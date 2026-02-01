/**
 * Medication List Panel Component
 * Displays current medications with drug interaction warnings
 */

import React, { useState } from "react";
import { Medication, DrugInteraction } from "../lib/api";

interface MedicationListPanelProps {
  patientId?: string;
  medications: Medication[];
  interactions?: DrugInteraction[];
  onAddMedication?: (medication: Medication) => void;
  onRefresh?: () => void;
}

export default function MedicationListPanel({
  patientId,
  medications,
  interactions,
  onAddMedication,
  onRefresh,
}: MedicationListPanelProps) {
  const [showInactiveopen, setShowInactive] = useState(false);
  const [selectedMed, setSelectedMed] = useState<Medication | null>(null);

  const activeMedications = medications.filter(
    (m) => m.status === "active" || !m.status
  );
  const inactiveMedications = medications.filter(
    (m) => m.status === "inactive" || m.status === "discontinued"
  );

  const displayedMeds = showInactive ? medications : activeMedications;

  // Get severe interactions count
  const severeInteractionsCount =
    interactions?.filter((i) => i.severity === "SEVERE").length || 0;

  const getDrugClassBadgeColor = (drugClass?: string) => {
    if (!drugClass) return "bg-gray-200 text-gray-700";
    if (drugClass.includes("TKI") || drugClass.includes("EGFR"))
      return "bg-blue-100 text-blue-700";
    if (drugClass.includes("immunotherapy") || drugClass.includes("PD"))
      return "bg-purple-100 text-purple-700";
    if (drugClass.includes("chemotherapy"))
      return "bg-green-100 text-green-700";
    return "bg-gray-200 text-gray-700";
  };

  if (!medications || medications.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          💊 Medications
        </h2>
        <p className="text-gray-500">No medications recorded</p>
        {onAddMedication && (
          <button
            onClick={() => {
              /* TODO: Open add medication modal */
            }}
            className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            + Add Medication
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          💊 Medications
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="ml-2 text-sm text-blue-600 hover:text-blue-800"
            >
              🔄 Refresh
            </button>
          )}
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setShowInactive(!showInactive)}
            className="px-3 py-1 text-sm rounded bg-gray-200 text-gray-700 hover:bg-gray-300"
          >
            {showInactive ? "Hide Inactive" : "Show Inactive"}
          </button>
          {onAddMedication && (
            <button
              onClick={() => {
                /* TODO: Open add medication modal */
              }}
              className="px-3 py-1 text-sm rounded bg-blue-600 text-white hover:bg-blue-700"
            >
              + Add
            </button>
          )}
        </div>
      </div>

      {/* Drug Interaction Warnings */}
      {severeInteractionsCount > 0 && (
        <div className="mb-4 bg-red-50 border-l-4 border-red-600 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <span className="text-2xl">⚠️</span>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Severe Drug Interactions Detected
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>
                  {severeInteractionsCount} severe drug interaction
                  {severeInteractionsCount > 1 ? "s" : ""} found. Review
                  interactions below.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Interactions List */}
      {interactions && interactions.length > 0 && (
        <div className="mb-4 space-y-2">
          <h3 className="text-sm font-semibold text-gray-700">
            Drug Interactions:
          </h3>
          {interactions.map((interaction, index) => (
            <div
              key={index}
              className={`p-3 rounded border ${
                interaction.severity === "SEVERE"
                  ? "bg-red-50 border-red-300"
                  : interaction.severity === "MODERATE"
                  ? "bg-yellow-50 border-yellow-300"
                  : "bg-blue-50 border-blue-300"
              }`}
            >
              <div className="flex items-start">
                <span className="mr-2">
                  {interaction.severity === "SEVERE"
                    ? "🔴"
                    : interaction.severity === "MODERATE"
                    ? "🟡"
                    : "🔵"}
                </span>
                <div className="flex-1">
                  <div className="font-medium text-sm">
                    {interaction.drug1} + {interaction.drug2}
                  </div>
                  <div className="text-sm text-gray-700 mt-1">
                    {interaction.clinical_effect}
                  </div>
                  {interaction.recommendation && (
                    <div className="text-sm text-gray-600 mt-1">
                      <strong>Recommendation:</strong>{" "}
                      {interaction.recommendation}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Medications Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Medication
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Dose
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Route
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Frequency
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Start Date
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {displayedMeds.map((med, index) => (
              <tr
                key={index}
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => setSelectedMed(med)}
              >
                <td className="px-4 py-3">
                  <div className="text-sm font-medium text-gray-900">
                    {med.drug_name || med.name || "Unknown"}
                  </div>
                  {med.drug_class && (
                    <span
                      className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getDrugClassBadgeColor(
                        med.drug_class
                      )}`}
                    >
                      {med.drug_class}
                    </span>
                  )}
                  {med.rxcui && (
                    <div className="text-xs text-gray-500 mt-1">
                      RxCUI: {med.rxcui}
                    </div>
                  )}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700">
                  {med.dose || "N/A"}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700">
                  {med.route || "N/A"}
                </td>
                <td className="px-4 py-3 text-sm text-gray-700">
                  {med.frequency || "N/A"}
                </td>
                <td className="px-4 py-3">
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${
                      med.status === "active"
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {med.status || "active"}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  {med.start_date
                    ? new Date(med.start_date).toLocaleDateString()
                    : "N/A"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {displayedMeds.length === 0 && (
        <p className="text-center text-gray-500 py-4">
          {showInactive ? "No medications found" : "No active medications"}
        </p>
      )}

      {/* Detail Modal */}
      {selectedMed && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setSelectedMed(null)}
        >
          <div
            className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-4">
              {selectedMed.drug_name || selectedMed.name} - Details
            </h3>
            <div className="space-y-3">
              <div>
                <strong>RxCUI:</strong> {selectedMed.rxcui || "Not specified"}
              </div>
              <div>
                <strong>Drug Class:</strong>{" "}
                {selectedMed.drug_class || "Not specified"}
              </div>
              <div>
                <strong>Dose:</strong> {selectedMed.dose || "Not specified"}
              </div>
              <div>
                <strong>Route:</strong> {selectedMed.route || "Not specified"}
              </div>
              <div>
                <strong>Frequency:</strong>{" "}
                {selectedMed.frequency || "Not specified"}
              </div>
              <div>
                <strong>Status:</strong> {selectedMed.status || "active"}
              </div>
              <div>
                <strong>Start Date:</strong>{" "}
                {selectedMed.start_date
                  ? new Date(selectedMed.start_date).toLocaleDateString()
                  : "Not specified"}
              </div>
              {selectedMed.end_date && (
                <div>
                  <strong>End Date:</strong>{" "}
                  {new Date(selectedMed.end_date).toLocaleDateString()}
                </div>
              )}
            </div>
            <button
              onClick={() => setSelectedMed(null)}
              className="mt-6 w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
