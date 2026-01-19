"use client";

import { useState } from "react";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Upload, FileJson, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

export default function FHIRUpload() {
  const { getAuthHeaders } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (selectedFile.type === "application/json") {
        setFile(selectedFile);
        setError("");
        setResult(null);
      } else {
        setError("Please select a valid JSON file (FHIR Bundle)");
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError("");
    setProgress(10);

    try {
      // Read file content
      const fileContent = await file.text();
      const bundle = JSON.parse(fileContent);

      setProgress(30);

      // Validate FHIR Bundle
      if (bundle.resourceType !== "Bundle") {
        throw new Error("Invalid FHIR Bundle: resourceType must be 'Bundle'");
      }

      setProgress(50);

      // Upload to API
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/fhir/import/bundle`, {
        method: "POST",
        headers: {
          ...getAuthHeaders(),
          "Content-Type": "application/json",
        },
        body: fileContent,
      });

      setProgress(80);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setProgress(100);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === "application/json") {
      setFile(droppedFile);
      setError("");
      setResult(null);
    } else {
      setError("Please drop a valid JSON file (FHIR Bundle)");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>FHIR Bundle Import</CardTitle>
          <CardDescription>
            Upload FHIR R4 Bundle to import patient data into the LCA system
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Upload Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              file
                ? "border-blue-500 bg-blue-50"
                : "border-gray-300 hover:border-gray-400"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <div className="flex flex-col items-center space-y-4">
              {file ? (
                <FileJson className="h-16 w-16 text-blue-600" />
              ) : (
                <Upload className="h-16 w-16 text-gray-400" />
              )}

              {file ? (
                <div>
                  <p className="text-lg font-semibold">{file.name}</p>
                  <p className="text-sm text-gray-600">
                    {(file.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              ) : (
                <>
                  <div>
                    <p className="text-lg font-semibold mb-2">
                      Drop FHIR Bundle here
                    </p>
                    <p className="text-sm text-gray-600">
                      or click to browse for a JSON file
                    </p>
                  </div>
                  <input
                    type="file"
                    accept="application/json,.json"
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload">
                    <Button variant="outline" asChild>
                      <span>Select File</span>
                    </Button>
                  </label>
                </>
              )}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <XCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Progress */}
          {uploading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Uploading...</span>
                <span className="text-sm text-gray-600">{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          )}

          {/* Result */}
          {result && (
            <Alert className="bg-green-50 border-green-200">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription>
                <div className="space-y-2">
                  <p className="font-semibold text-green-900">
                    Upload Successful!
                  </p>
                  <div className="text-sm text-green-800">
                    <p>• Imported {result.total_resources} resources</p>
                    <p>• Created {result.created_patients} patient records</p>
                    {result.patient_ids && (
                      <p>
                        • Patient IDs:{" "}
                        {result.patient_ids.slice(0, 3).join(", ")}
                        {result.patient_ids.length > 3 &&
                          ` and ${result.patient_ids.length - 3} more...`}
                      </p>
                    )}
                  </div>
                </div>
              </AlertDescription>
            </Alert>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="flex-1"
            >
              {uploading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Upload Bundle
                </>
              )}
            </Button>

            {file && !uploading && (
              <Button
                variant="outline"
                onClick={() => {
                  setFile(null);
                  setResult(null);
                  setError("");
                }}
              >
                Clear
              </Button>
            )}
          </div>

          {/* FHIR Bundle Example */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-2">FHIR Bundle Example:</h3>
            <pre className="text-xs overflow-auto bg-white p-3 rounded border">
{`{
  "resourceType": "Bundle",
  "type": "transaction",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "patient-123",
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1955-01-15"
      },
      "request": {
        "method": "POST",
        "url": "Patient"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "subject": {"reference": "Patient/patient-123"},
        "code": {
          "coding": [{
            "system": "http://snomed.info/sct",
            "code": "254637007",
            "display": "Non-small cell lung cancer"
          }]
        }
      },
      "request": {
        "method": "POST",
        "url": "Condition"
      }
    }
  ]
}`}
            </pre>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
