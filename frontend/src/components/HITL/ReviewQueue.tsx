"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  User,
  Calendar,
  TrendingUp,
  Filter
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

interface ReviewCase {
  case_id: string;
  patient_id: string;
  case_type: string;
  recommendations: any;
  confidence_score: number;
  reason: string;
  priority: number;
  status: string;
  submitted_at: string;
  submitted_by: string;
  metadata?: any;
}

interface QueueSummary {
  total_pending: number;
  high_priority: number;
  medium_priority: number;
  low_priority: number;
  avg_waiting_time_hours: number;
  oldest_case_hours: number;
}

export default function ReviewQueue() {
  const { getAuthHeaders } = useAuth();
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [summary, setSummary] = useState<QueueSummary | null>(null);
  const [selectedCase, setSelectedCase] = useState<ReviewCase | null>(null);
  const [loading, setLoading] = useState(true);
  const [reviewAction, setReviewAction] = useState<"approve" | "reject" | null>(null);
  const [feedback, setFeedback] = useState("");
  const [filterStatus, setFilterStatus] = useState("pending");
  const [filterPriority, setFilterPriority] = useState<number | null>(null);

  useEffect(() => {
    fetchQueue();
    fetchSummary();
  }, [filterStatus, filterPriority]);

  const fetchQueue = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const params = new URLSearchParams();
      
      if (filterStatus) params.append("status_filter", filterStatus);
      if (filterPriority) params.append("priority", filterPriority.toString());
      
      const response = await fetch(
        `${apiUrl}/api/v1/hitl/queue?${params.toString()}`,
        {
          headers: getAuthHeaders(),
        }
      );

      if (response.ok) {
        const data = await response.json();
        setCases(data);
      }
    } catch (error) {
      console.error("Failed to fetch queue:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/hitl/queue/summary`, {
        headers: getAuthHeaders(),
      });

      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      }
    } catch (error) {
      console.error("Failed to fetch summary:", error);
    }
  };

  const handleReview = async () => {
    if (!selectedCase || !reviewAction || !feedback) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(
        `${apiUrl}/api/v1/hitl/cases/${selectedCase.case_id}/review`,
        {
          method: "POST",
          headers: {
            ...getAuthHeaders(),
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            action: reviewAction,
            feedback: feedback,
          }),
        }
      );

      if (response.ok) {
        setSelectedCase(null);
        setFeedback("");
        setReviewAction(null);
        fetchQueue();
        fetchSummary();
      }
    } catch (error) {
      console.error("Review submission failed:", error);
    }
  };

  const getPriorityBadge = (priority: number) => {
    const variants = {
      1: { color: "bg-red-100 text-red-800", label: "High" },
      2: { color: "bg-yellow-100 text-yellow-800", label: "Medium" },
      3: { color: "bg-blue-100 text-blue-800", label: "Low" },
    };
    const variant = variants[priority as keyof typeof variants] || variants[2];
    return (
      <Badge className={variant.color}>
        {variant.label} Priority
      </Badge>
    );
  };

  const getStatusIcon = (status: string) => {
    const icons = {
      pending: <Clock className="h-4 w-4 text-yellow-600" />,
      approved: <CheckCircle className="h-4 w-4 text-green-600" />,
      rejected: <XCircle className="h-4 w-4 text-red-600" />,
    };
    return icons[status as keyof typeof icons] || icons.pending;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Pending Cases</p>
                <p className="text-3xl font-bold">{summary?.total_pending || 0}</p>
              </div>
              <AlertCircle className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">High Priority</p>
                <p className="text-3xl font-bold text-red-600">
                  {summary?.high_priority || 0}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-red-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Wait Time</p>
                <p className="text-3xl font-bold">
                  {summary?.avg_waiting_time_hours.toFixed(1) || 0}h
                </p>
              </div>
              <Clock className="h-8 w-8 text-yellow-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Oldest Case</p>
                <p className="text-3xl font-bold">
                  {summary?.oldest_case_hours.toFixed(1) || 0}h
                </p>
              </div>
              <Calendar className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex gap-4 items-center">
            <Filter className="h-5 w-5 text-gray-600" />
            <div className="flex gap-2">
              <Button
                variant={filterStatus === "pending" ? "default" : "outline"}
                onClick={() => setFilterStatus("pending")}
                size="sm"
              >
                Pending
              </Button>
              <Button
                variant={filterStatus === "approved" ? "default" : "outline"}
                onClick={() => setFilterStatus("approved")}
                size="sm"
              >
                Approved
              </Button>
              <Button
                variant={filterStatus === "rejected" ? "default" : "outline"}
                onClick={() => setFilterStatus("rejected")}
                size="sm"
              >
                Rejected
              </Button>
            </div>
            <div className="border-l pl-4 flex gap-2">
              <Button
                variant={filterPriority === 1 ? "destructive" : "outline"}
                onClick={() => setFilterPriority(filterPriority === 1 ? null : 1)}
                size="sm"
              >
                High Priority
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Cases List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Review Queue</h2>
          {loading ? (
            <p>Loading cases...</p>
          ) : cases.length === 0 ? (
            <Alert>
              <AlertDescription>No cases found matching filters</AlertDescription>
            </Alert>
          ) : (
            cases.map((case_) => (
              <Card
                key={case_.case_id}
                className={`cursor-pointer transition-all ${
                  selectedCase?.case_id === case_.case_id
                    ? "border-blue-500 border-2"
                    : ""
                }`}
                onClick={() => setSelectedCase(case_)}
              >
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-lg">
                        Patient {case_.patient_id}
                      </CardTitle>
                      <CardDescription>{case_.case_type}</CardDescription>
                    </div>
                    <div className="flex flex-col gap-2 items-end">
                      {getPriorityBadge(case_.priority)}
                      <div className="flex items-center gap-1">
                        {getStatusIcon(case_.status)}
                        <span className="text-sm capitalize">{case_.status}</span>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <User className="h-4 w-4 text-gray-500" />
                      <span className="text-gray-600">
                        Submitted by {case_.submitted_by}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-gray-500" />
                      <span className="text-gray-600">
                        {new Date(case_.submitted_at).toLocaleString()}
                      </span>
                    </div>
                    <div className="mt-3 p-3 bg-gray-50 rounded">
                      <p className="font-semibold mb-1">Reason for Review:</p>
                      <p className="text-gray-700">{case_.reason}</p>
                    </div>
                    <div className="mt-2">
                      <p className="font-semibold">Confidence Score:</p>
                      <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                        <div
                          className={`h-2 rounded-full ${
                            case_.confidence_score > 0.8
                              ? "bg-green-600"
                              : case_.confidence_score > 0.6
                              ? "bg-yellow-600"
                              : "bg-red-600"
                          }`}
                          style={{ width: `${case_.confidence_score * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-gray-600 mt-1">
                        {(case_.confidence_score * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>

        {/* Review Panel */}
        <div>
          <h2 className="text-xl font-bold mb-4">Review Details</h2>
          {selectedCase ? (
            <Card>
              <CardHeader>
                <CardTitle>Case {selectedCase.case_id}</CardTitle>
                <CardDescription>
                  Patient {selectedCase.patient_id}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Recommendations:</h3>
                  <pre className="bg-gray-50 p-3 rounded text-sm overflow-auto">
                    {JSON.stringify(selectedCase.recommendations, null, 2)}
                  </pre>
                </div>

                {selectedCase.metadata && (
                  <div>
                    <h3 className="font-semibold mb-2">Clinical Context:</h3>
                    <pre className="bg-gray-50 p-3 rounded text-sm overflow-auto">
                      {JSON.stringify(selectedCase.metadata, null, 2)}
                    </pre>
                  </div>
                )}

                {selectedCase.status === "pending" && (
                  <>
                    <div className="space-y-2">
                      <label className="font-semibold">Clinical Feedback:</label>
                      <Textarea
                        placeholder="Provide your clinical reasoning (minimum 10 characters)..."
                        value={feedback}
                        onChange={(e) => setFeedback(e.target.value)}
                        rows={5}
                        minLength={10}
                      />
                      <p className="text-xs text-gray-500">
                        {feedback.length}/10 minimum characters
                      </p>
                    </div>

                    <div className="flex gap-3">
                      <Button
                        onClick={() => {
                          setReviewAction("approve");
                          handleReview();
                        }}
                        disabled={feedback.length < 10}
                        className="flex-1 bg-green-600 hover:bg-green-700"
                      >
                        <CheckCircle className="mr-2 h-4 w-4" />
                        Approve
                      </Button>
                      <Button
                        onClick={() => {
                          setReviewAction("reject");
                          handleReview();
                        }}
                        disabled={feedback.length < 10}
                        className="flex-1 bg-red-600 hover:bg-red-700"
                      >
                        <XCircle className="mr-2 h-4 w-4" />
                        Reject
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          ) : (
            <Alert>
              <AlertDescription>
                Select a case from the queue to review
              </AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    </div>
  );
}
