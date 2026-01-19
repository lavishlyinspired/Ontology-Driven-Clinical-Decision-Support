"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Download,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

interface BatchJob {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  total_items: number;
  processed_items: number;
  failed_items: number;
  created_at: string;
  completed_at?: string;
  error?: string;
}

export default function BatchJobMonitor() {
  const { getAuthHeaders } = useAuth();
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchJobs = async () => {
    setLoading(true);
    setError("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/batch/jobs`, {
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch jobs");
      }

      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const downloadResults = async (jobId: string) => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(
        `${apiUrl}/api/v1/batch/jobs/${jobId}/results`,
        {
          headers: getAuthHeaders(),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to download results");
      }

      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch-results-${jobId}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert("Failed to download results");
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-red-600" />;
      case "processing":
        return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;
      default:
        return <Clock className="h-5 w-5 text-gray-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-100 text-green-800 border-green-300";
      case "failed":
        return "bg-red-100 text-red-800 border-red-300";
      case "processing":
        return "bg-blue-100 text-blue-800 border-blue-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const calculateProgress = (job: BatchJob) => {
    if (job.total_items === 0) return 0;
    return Math.round((job.processed_items / job.total_items) * 100);
  };

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const diff = Math.floor((end.getTime() - start.getTime()) / 1000);

    const hours = Math.floor(diff / 3600);
    const minutes = Math.floor((diff % 3600) / 60);
    const seconds = diff % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Batch Job Monitor</CardTitle>
              <CardDescription>
                Track population-level analysis and bulk import jobs
              </CardDescription>
            </div>
            <Button onClick={fetchJobs} disabled={loading}>
              <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {loading && jobs.length === 0 ? (
            <div className="text-center py-8">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600">Loading jobs...</p>
            </div>
          ) : jobs.length === 0 ? (
            <Alert>
              <AlertDescription>
                No batch jobs found. Submit a batch processing job to see it here.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="space-y-4">
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="border rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(job.status)}
                      <div>
                        <p className="font-semibold">Job ID: {job.job_id}</p>
                        <p className="text-sm text-gray-600">
                          Started: {new Date(job.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <Badge className={getStatusColor(job.status)}>
                      {job.status.toUpperCase()}
                    </Badge>
                  </div>

                  {/* Progress */}
                  {job.status === "processing" && (
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-2">
                        <span>
                          Progress: {job.processed_items} / {job.total_items}
                        </span>
                        <span>{calculateProgress(job)}%</span>
                      </div>
                      <Progress value={calculateProgress(job)} />
                    </div>
                  )}

                  {/* Stats */}
                  <div className="grid grid-cols-4 gap-4 mb-4">
                    <div className="text-center p-3 bg-gray-50 rounded">
                      <p className="text-2xl font-bold">{job.total_items}</p>
                      <p className="text-xs text-gray-600">Total Items</p>
                    </div>
                    <div className="text-center p-3 bg-blue-50 rounded">
                      <p className="text-2xl font-bold text-blue-600">
                        {job.processed_items}
                      </p>
                      <p className="text-xs text-gray-600">Processed</p>
                    </div>
                    <div className="text-center p-3 bg-red-50 rounded">
                      <p className="text-2xl font-bold text-red-600">
                        {job.failed_items}
                      </p>
                      <p className="text-xs text-gray-600">Failed</p>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded">
                      <p className="text-2xl font-bold text-green-600">
                        {job.processed_items - job.failed_items}
                      </p>
                      <p className="text-xs text-gray-600">Success</p>
                    </div>
                  </div>

                  {/* Duration */}
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-600">
                      <Clock className="inline h-4 w-4 mr-1" />
                      Duration:{" "}
                      {formatDuration(job.created_at, job.completed_at)}
                    </p>

                    {/* Actions */}
                    {job.status === "completed" && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => downloadResults(job.job_id)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download Results
                      </Button>
                    )}
                  </div>

                  {/* Error */}
                  {job.error && (
                    <Alert variant="destructive" className="mt-4">
                      <XCircle className="h-4 w-4" />
                      <AlertDescription>{job.error}</AlertDescription>
                    </Alert>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
