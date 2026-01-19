"""
Batch Processing Service

Enables queued batch processing for population-level analysis.
Uses Celery for distributed task execution.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import uuid
import json


class TaskStatus(str, Enum):
    """Status of a batch task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class BatchTask(BaseModel):
    """A batch processing task."""
    task_id: str
    task_type: str
    
    # Task configuration
    input_data: Dict[str, Any]
    parameters: Dict[str, Any] = {}
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Progress tracking
    progress: float = 0.0
    current_step: Optional[str] = None
    total_items: Optional[int] = None
    processed_items: int = 0
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    queued_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    created_by: Optional[str] = None
    tags: List[str] = []


class BatchJob(BaseModel):
    """A collection of related batch tasks."""
    job_id: str
    job_name: str
    description: Optional[str] = None
    
    # Tasks in this job
    task_ids: List[str] = []
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Progress (aggregated from tasks)
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    results_summary: Dict[str, Any] = {}
    
    # Configuration
    created_by: Optional[str] = None


class BatchProcessor:
    """Service for batch processing and queue management."""
    
    def __init__(self):
        """Initialize batch processor."""
        self.tasks: Dict[str, BatchTask] = {}
        self.jobs: Dict[str, BatchJob] = {}
        
        # Task queues by priority
        self.queues: Dict[TaskPriority, List[str]] = {
            TaskPriority.URGENT: [],
            TaskPriority.HIGH: [],
            TaskPriority.NORMAL: [],
            TaskPriority.LOW: []
        }
        
        # Task handlers
        self.task_handlers: Dict[str, Callable] = {}
        
        # Worker pool (simplified - in production use Celery)
        self._workers_running = False
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        print(f"📝 Registered handler for task type: {task_type}")
    
    def create_task(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> BatchTask:
        """
        Create a new batch task.
        
        Args:
            task_type: Type of task (e.g., 'population_analysis', 'batch_prediction')
            input_data: Input data for the task
            parameters: Additional parameters
            priority: Task priority
            created_by: User who created the task
            tags: Tags for organization
        
        Returns:
            Created BatchTask
        """
        task_id = str(uuid.uuid4())
        
        task = BatchTask(
            task_id=task_id,
            task_type=task_type,
            input_data=input_data,
            parameters=parameters or {},
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            created_by=created_by,
            tags=tags or []
        )
        
        self.tasks[task_id] = task
        
        print(f"📋 Created task {task_id}: {task_type} ({priority.value} priority)")
        
        return task
    
    def queue_task(self, task_id: str) -> bool:
        """Add a task to the processing queue."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.PENDING:
            print(f"⚠️ Task {task_id} already queued/processed")
            return False
        
        # Add to appropriate queue
        self.queues[task.priority].append(task_id)
        
        # Update status
        task.status = TaskStatus.QUEUED
        task.queued_at = datetime.now()
        
        print(f"📥 Queued task {task_id} ({task.priority.value} priority)")
        
        # Start workers if not running
        if not self._workers_running:
            self._start_workers()
        
        return True
    
    def create_job(
        self,
        job_name: str,
        task_specs: List[Dict[str, Any]],
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> BatchJob:
        """
        Create a batch job with multiple tasks.
        
        Args:
            job_name: Name of the job
            task_specs: List of task specifications (each with task_type, input_data, etc.)
            description: Job description
            created_by: User who created the job
        
        Returns:
            Created BatchJob
        """
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            job_name=job_name,
            description=description,
            total_tasks=len(task_specs),
            created_at=datetime.now(),
            created_by=created_by
        )
        
        # Create all tasks
        for spec in task_specs:
            task = self.create_task(
                task_type=spec['task_type'],
                input_data=spec['input_data'],
                parameters=spec.get('parameters'),
                priority=spec.get('priority', TaskPriority.NORMAL),
                created_by=created_by,
                tags=[f"job:{job_id}"]
            )
            
            job.task_ids.append(task.task_id)
            
            # Queue the task
            self.queue_task(task.task_id)
        
        self.jobs[job_id] = job
        
        print(f"📦 Created job {job_id}: {job_name} ({len(task_specs)} tasks)")
        
        return job
    
    def get_task(self, task_id: str) -> Optional[BatchTask]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or queued task."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        if task.status == TaskStatus.RUNNING:
            print(f"⚠️ Cannot cancel running task {task_id}")
            return False
        
        # Remove from queue
        for queue in self.queues.values():
            if task_id in queue:
                queue.remove(task_id)
        
        # Update status
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        print(f"🚫 Cancelled task {task_id}")
        
        return True
    
    def _start_workers(self):
        """Start background workers (simplified)."""
        # In production, this would start Celery workers
        # For now, we'll process synchronously
        self._workers_running = True
        print("👷 Workers started")
    
    async def process_next_task(self) -> Optional[str]:
        """
        Process the next task in the queue.
        
        Returns task_id of processed task, or None if queue empty.
        """
        # Find highest priority non-empty queue
        for priority in [TaskPriority.URGENT, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self.queues[priority]
            
            if queue:
                task_id = queue.pop(0)
                await self._execute_task(task_id)
                return task_id
        
        return None
    
    async def _execute_task(self, task_id: str):
        """Execute a single task."""
        task = self.tasks[task_id]
        
        # Update status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        print(f"▶️ Executing task {task_id}: {task.task_type}")
        
        try:
            # Get handler
            if task.task_type not in self.task_handlers:
                raise ValueError(f"No handler registered for task type: {task.task_type}")
            
            handler = self.task_handlers[task.task_type]
            
            # Execute handler
            result = await handler(task.input_data, task.parameters, task)
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 1.0
            task.completed_at = datetime.now()
            
            print(f"✅ Task {task_id} completed")
            
            # Update job if this task belongs to one
            self._update_job_progress(task_id)
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            print(f"❌ Task {task_id} failed: {e}")
            
            # Update job
            self._update_job_progress(task_id)
    
    def _update_job_progress(self, task_id: str):
        """Update job progress when a task completes."""
        # Find job containing this task
        for job in self.jobs.values():
            if task_id in job.task_ids:
                # Count completed and failed tasks
                completed = sum(
                    1 for tid in job.task_ids
                    if self.tasks[tid].status == TaskStatus.COMPLETED
                )
                failed = sum(
                    1 for tid in job.task_ids
                    if self.tasks[tid].status == TaskStatus.FAILED
                )
                
                job.completed_tasks = completed
                job.failed_tasks = failed
                
                # Update job status
                if completed + failed == job.total_tasks:
                    job.status = TaskStatus.COMPLETED
                    job.completed_at = datetime.now()
                    
                    print(f"✅ Job {job.job_id} completed: {completed}/{job.total_tasks} successful")
                
                break
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about task queues."""
        return {
            'total_tasks': len(self.tasks),
            'queued_tasks': sum(len(q) for q in self.queues.values()),
            'by_priority': {
                priority.value: len(queue)
                for priority, queue in self.queues.items()
            },
            'by_status': {
                status.value: len([t for t in self.tasks.values() if t.status == status])
                for status in TaskStatus
            },
            'total_jobs': len(self.jobs),
            'active_jobs': len([j for j in self.jobs.values() if j.status == TaskStatus.RUNNING])
        }
    
    def export_results(
        self,
        job_id: str,
        format: str = 'json'
    ) -> str:
        """
        Export job results.
        
        Args:
            job_id: Job ID to export
            format: Export format ('json' or 'csv')
        
        Returns:
            Serialized results
        """
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        # Collect all task results
        results = []
        for task_id in job.task_ids:
            task = self.tasks[task_id]
            results.append({
                'task_id': task_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'result': task.result,
                'error': task.error,
                'duration_seconds': (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.started_at and task.completed_at
                    else None
                )
            })
        
        if format == 'json':
            return json.dumps({
                'job_id': job_id,
                'job_name': job.job_name,
                'total_tasks': job.total_tasks,
                'completed_tasks': job.completed_tasks,
                'failed_tasks': job.failed_tasks,
                'results': results
            }, indent=2, default=str)
        
        elif format == 'csv':
            # Simple CSV export
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=['task_id', 'task_type', 'status', 'duration_seconds', 'error']
            )
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'task_id': result['task_id'],
                    'task_type': result['task_type'],
                    'status': result['status'],
                    'duration_seconds': result['duration_seconds'],
                    'error': result.get('error', '')
                })
            
            return output.getvalue()


# Global batch processor instance
batch_processor = BatchProcessor()


# Example task handlers

async def population_analysis_handler(
    input_data: Dict[str, Any],
    parameters: Dict[str, Any],
    task: BatchTask
) -> Dict[str, Any]:
    """
    Handler for population-level analysis.
    
    Analyzes multiple patients and generates aggregate statistics.
    """
    from ..agents.integrated_workflow import integrated_workflow
    
    patient_ids = input_data.get('patient_ids', [])
    task.total_items = len(patient_ids)
    
    results = []
    
    for i, patient_id in enumerate(patient_ids):
        # Update progress
        task.processed_items = i
        task.progress = i / len(patient_ids)
        task.current_step = f"Analyzing patient {patient_id}"
        
        # Run analysis
        try:
            # Get patient data (simplified - would load from DB)
            patient_data = {'patient_id': patient_id}
            
            # Run integrated workflow
            result = await integrated_workflow.run_workflow(patient_data)
            
            results.append({
                'patient_id': patient_id,
                'success': True,
                'recommendation': result.get('recommendation'),
                'confidence': result.get('confidence')
            })
        
        except Exception as e:
            results.append({
                'patient_id': patient_id,
                'success': False,
                'error': str(e)
            })
    
    # Calculate aggregate statistics
    successful = [r for r in results if r['success']]
    
    aggregate_stats = {
        'total_patients': len(patient_ids),
        'successful_analyses': len(successful),
        'failed_analyses': len(results) - len(successful),
        'average_confidence': (
            sum(r['confidence'] for r in successful) / len(successful)
            if successful else 0
        ),
        'results': results
    }
    
    return aggregate_stats


# Register default handlers
batch_processor.register_task_handler('population_analysis', population_analysis_handler)


# Convenience functions

def create_population_analysis_job(
    patient_ids: List[str],
    job_name: str = "Population Analysis",
    created_by: Optional[str] = None,
    batch_size: int = 10
) -> BatchJob:
    """
    Create a population analysis job.
    
    Args:
        patient_ids: List of patient IDs to analyze
        job_name: Name for the job
        created_by: User creating the job
        batch_size: Number of patients per task
    
    Returns:
        Created BatchJob
    """
    # Split into batches
    batches = [
        patient_ids[i:i+batch_size]
        for i in range(0, len(patient_ids), batch_size)
    ]
    
    # Create task specs
    task_specs = [
        {
            'task_type': 'population_analysis',
            'input_data': {'patient_ids': batch},
            'priority': TaskPriority.NORMAL
        }
        for batch in batches
    ]
    
    # Create job
    return batch_processor.create_job(
        job_name=job_name,
        task_specs=task_specs,
        description=f"Population analysis for {len(patient_ids)} patients",
        created_by=created_by
    )
