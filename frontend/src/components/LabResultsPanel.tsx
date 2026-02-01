/**
 * Lab Results Panel Component
 * Displays laboratory results with interpretations, severity color-coding, and filtering
 */

import React, { useState, useMemo } from "react";
import { LabResult, LabInterpretation } from "../lib/api";

interface LabResultsPanelProps {
  patientId?: string;
  labResults: LabResult[];
  labInterpretations?: LabInterpretation[];
  onRefresh?: () => void;
}

export default function LabResultsPanel({
  patientId,
  labResults,
  labInterpretations,
  onRefresh,
}: LabResultsPanelProps) {
  const [filter, setFilter] = useState<"all" | "abnormal" | "critical">("all");
  const [selectedLab, setSelectedLab] = useState<LabResult | null>(null);

  // Merge lab results with interpretations
  const enhancedResults = useMemo(() => {
    if (!labInterpretations || labInterpretations.length === 0) {
      return labResults;
    }

    return labResults.map((lab) => {
      const interp = labInterpretations.find(
        (i) => i.loinc_code === lab.loinc_code
      );
      return interp || lab;
    });
  }, [labResults, labInterpretations]);

  // Filter results
  const filteredResults = useMemo(() => {
    if (filter === "all") return enhancedResults;
    if (filter === "abnormal")
      return enhancedResults.filter((r) => r.interpretation !== "normal");
    if (filter === "critical")
      return enhancedResults.filter(
        (r) =>
          r.severity === "grade3" ||
          r.severity === "grade4" ||
          r.severity === "critical"
      );
    return enhancedResults;
  }, [enhancedResults, filter]);

  const getSeverityColor = (severity?: string, interpretation?: string) => {
    if (severity === "grade4" || severity === "critical")
      return "text-red-700 bg-red-100";
    if (severity === "grade3") return "text-red-600 bg-red-50";
    if (severity === "grade2") return "text-yellow-600 bg-yellow-50";
    if (severity === "grade1") return "text-yellow-500 bg-yellow-50";
    if (interpretation === "high" || interpretation === "low")
      return "text-orange-600 bg-orange-50";
    return "text-green-700 bg-green-100";
  };

  const getSeverityIcon = (severity?: string, interpretation?: string) => {
    if (severity === "grade4" || severity === "critical") return "🔴";
    if (severity === "grade3") return "🟠";
    if (severity === "grade2" || severity === "grade1") return "🟡";
    if (interpretation === "high" || interpretation === "low") return "⚠️";
    return "✅";
  };

  if (!labResults || labResults.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          🧪 Laboratory Results
        </h2>
        <p className="text-gray-500">No laboratory results available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          🧪 Laboratory Results
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
            onClick={() => setFilter("all")}
            className={`px-3 py-1 rounded ${
              filter === "all"
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilter("abnormal")}
            className={`px-3 py-1 rounded ${
              filter === "abnormal"
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
          >
            Abnormal
          </button>
          <button
            onClick={() => setFilter("critical")}
            className={`px-3 py-1 rounded ${
              filter === "critical"
                ? "bg-blue-600 text-white"
                : "bg-gray-200 text-gray-700"
            }`}
          >
            Critical
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Test Name
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Value
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Reference Range
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Interpretation
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                Date
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {filteredResults.map((lab, index) => (
              <tr
                key={index}
                className="hover:bg-gray-50 cursor-pointer"
                onClick={() => setSelectedLab(lab)}
              >
                <td className="px-4 py-3">
                  <span className="text-xl">
                    {getSeverityIcon(lab.severity, lab.interpretation)}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="text-sm font-medium text-gray-900">
                    {lab.loinc_name || lab.test_name || "Unknown Test"}
                  </div>
                  <div className="text-xs text-gray-500">{lab.loinc_code}</div>
                </td>
                <td className="px-4 py-3">
                  <span className="text-sm font-semibold">
                    {lab.value} {lab.units || lab.unit || ""}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  {lab.reference_range || "N/A"}
                </td>
                <td className="px-4 py-3">
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${getSeverityColor(
                      lab.severity,
                      lab.interpretation
                    )}`}
                  >
                    {lab.interpretation || "Unknown"}
                    {lab.severity && lab.severity !== "normal" && (
                      <span className="ml-1">({lab.severity})</span>
                    )}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  {lab.test_date
                    ? new Date(lab.test_date).toLocaleDateString()
                    : "N/A"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detail Modal */}
      {selectedLab && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
          onClick={() => setSelectedLab(null)}
        >
          <div
            className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-semibold mb-4">
              {selectedLab.loinc_name || selectedLab.test_name} - Detailed
              Interpretation
            </h3>
            <div className="space-y-3">
              <div>
                <strong>LOINC Code:</strong> {selectedLab.loinc_code}
              </div>
              <div>
                <strong>Value:</strong> {selectedLab.value}{" "}
                {selectedLab.units || selectedLab.unit}
              </div>
              <div>
                <strong>Reference Range:</strong>{" "}
                {selectedLab.reference_range || "Not specified"}
              </div>
              <div>
                <strong>Interpretation:</strong>{" "}
                <span
                  className={`px-2 py-1 text-xs font-semibold rounded ${getSeverityColor(
                    selectedLab.severity,
                    selectedLab.interpretation
                  )}`}
                >
                  {selectedLab.interpretation}
                </span>
              </div>
              <div>
                <strong>Severity:</strong> {selectedLab.severity || "Normal"}
              </div>
              {"clinical_significance" in selectedLab &&
                (selectedLab as LabInterpretation).clinical_significance && (
                  <div>
                    <strong>Clinical Significance:</strong>
                    <p className="mt-1 text-gray-700">
                      {(selectedLab as LabInterpretation).clinical_significance}
                    </p>
                  </div>
                )}
              {"recommendation" in selectedLab &&
                (selectedLab as LabInterpretation).recommendation && (
                  <div>
                    <strong>Recommendation:</strong>
                    <p className="mt-1 text-gray-700">
                      {(selectedLab as LabInterpretation).recommendation}
                    </p>
                  </div>
                )}
            </div>
            <button
              onClick={() => setSelectedLab(null)}
              className="mt-6 w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {filteredResults.length === 0 && (
        <p className="text-center text-gray-500 py-4">
          No results match the selected filter
        </p>
      )}
    </div>
  );
}
