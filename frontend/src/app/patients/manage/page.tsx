"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";

type PatientData = {
  patient_id: string;
  name: string;
  demographics: {
    age: number;
    sex: string;
    ethnicity?: string;
  };
  clinical_data: {
    tnm_stage: string;
    histology_type: string;
    performance_status: number;
    fev1_percent?: number;
    laterality?: string;
    diagnosis?: string;
  };
  biomarkers?: {
    egfr_mutation?: string;
    alk_rearrangement?: boolean;
    pdl1_tps?: number;
  };
  comorbidities?: string[];
};

export default function PatientManagementPage() {
  const [loading, setLoading] = useState(false);
  const [patient, setPatient] = useState<PatientData>({
    patient_id: "",
    name: "",
    demographics: {
      age: 65,
      sex: "M",
    },
    clinical_data: {
      tnm_stage: "IIIA",
      histology_type: "Adenocarcinoma",
      performance_status: 1,
    },
  });

  const handleCreatePatient = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/v1/patients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      });

      if (!response.ok) {
        throw new Error(`Failed to create patient: ${response.statusText}`);
      }

      const result = await response.json();
      toast.success(`Patient created successfully! ID: ${result.patient_id}`);
      
      // Reset form
      setPatient({
        patient_id: "",
        name: "",
        demographics: { age: 65, sex: "M" },
        clinical_data: {
          tnm_stage: "IIIA",
          histology_type: "Adenocarcinoma",
          performance_status: 1,
        },
      });
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Failed to create patient");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadPatient = async (patientId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/patients/${patientId}`);
      
      if (!response.ok) {
        throw new Error(`Patient not found: ${patientId}`);
      }

      const data = await response.json();
      setPatient(data);
      toast.success("Patient loaded successfully");
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Failed to load patient");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePatient = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/patients/${patient.patient_id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          demographics: patient.demographics,
          clinical_data: patient.clinical_data,
          biomarkers: patient.biomarkers,
          comorbidities: patient.comorbidities,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to update patient");
      }

      toast.success("Patient updated successfully");
    } catch (error) {
      console.error(error);
      toast.error(error instanceof Error ? error.message : "Failed to update patient");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Patient Management</h1>
          <p className="text-muted-foreground">Create, update, and manage patient records</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Patient Information Card */}
        <Card>
          <CardHeader>
            <CardTitle>Patient Information</CardTitle>
            <CardDescription>Basic demographics and identification</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="patient_id">Patient ID</Label>
              <Input
                id="patient_id"
                value={patient.patient_id}
                onChange={(e) => setPatient({ ...patient, patient_id: e.target.value })}
                placeholder="PAT-12345"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={patient.name}
                onChange={(e) => setPatient({ ...patient, name: e.target.value })}
                placeholder="John Doe"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  value={patient.demographics.age}
                  onChange={(e) =>
                    setPatient({
                      ...patient,
                      demographics: { ...patient.demographics, age: parseInt(e.target.value) },
                    })
                  }
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="sex">Sex</Label>
                <Select
                  value={patient.demographics.sex}
                  onValueChange={(value) =>
                    setPatient({
                      ...patient,
                      demographics: { ...patient.demographics, sex: value },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="M">Male</SelectItem>
                    <SelectItem value="F">Female</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Clinical Data Card */}
        <Card>
          <CardHeader>
            <CardTitle>Clinical Data</CardTitle>
            <CardDescription>Disease staging and clinical characteristics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="tnm_stage">TNM Stage</Label>
              <Select
                value={patient.clinical_data.tnm_stage}
                onValueChange={(value) =>
                  setPatient({
                    ...patient,
                    clinical_data: { ...patient.clinical_data, tnm_stage: value },
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="IA">IA</SelectItem>
                  <SelectItem value="IB">IB</SelectItem>
                  <SelectItem value="IIA">IIA</SelectItem>
                  <SelectItem value="IIB">IIB</SelectItem>
                  <SelectItem value="IIIA">IIIA</SelectItem>
                  <SelectItem value="IIIB">IIIB</SelectItem>
                  <SelectItem value="IV">IV</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="histology">Histology Type</Label>
              <Select
                value={patient.clinical_data.histology_type}
                onValueChange={(value) =>
                  setPatient({
                    ...patient,
                    clinical_data: { ...patient.clinical_data, histology_type: value },
                  })
                }
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Adenocarcinoma">Adenocarcinoma</SelectItem>
                  <SelectItem value="Squamous cell carcinoma">Squamous Cell Carcinoma</SelectItem>
                  <SelectItem value="Small cell carcinoma">Small Cell Carcinoma</SelectItem>
                  <SelectItem value="Large cell carcinoma">Large Cell Carcinoma</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="performance_status">Performance Status (0-4)</Label>
              <Input
                id="performance_status"
                type="number"
                min="0"
                max="4"
                value={patient.clinical_data.performance_status}
                onChange={(e) =>
                  setPatient({
                    ...patient,
                    clinical_data: {
                      ...patient.clinical_data,
                      performance_status: parseInt(e.target.value),
                    },
                  })
                }
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <Button onClick={handleCreatePatient} disabled={loading || !patient.patient_id}>
          {loading ? "Creating..." : "Create Patient"}
        </Button>
        
        <Button onClick={handleUpdatePatient} disabled={loading || !patient.patient_id} variant="outline">
          {loading ? "Updating..." : "Update Patient"}
        </Button>
        
        <Button
          onClick={() => {
            const id = prompt("Enter patient ID to load:");
            if (id) handleLoadPatient(id);
          }}
          disabled={loading}
          variant="secondary"
        >
          Load Patient
        </Button>
      </div>
    </div>
  );
}
