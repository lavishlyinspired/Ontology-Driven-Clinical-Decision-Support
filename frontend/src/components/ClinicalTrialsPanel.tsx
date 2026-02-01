/**
 * Clinical Trials Panel Component
 * Displays eligible clinical trials sorted by match score
 */

import React, { useState } from "react";
import { ClinicalTrial } from "../lib/api";

interface ClinicalTrialsPanelProps {
  patientId?: string;
  trials: ClinicalTrial[];
  onEnroll?: (nctId: string) => void;
}

export default function ClinicalTrialsPanel({
  patientId,
  trials,
  onEnroll,
}: ClinicalTrialsPanelProps) {
  const [expandedTrial, setExpandedTrial] = useState<string | null>(null);

  const sortedTrials = [...trials].sort((a, b) => {
    const scoreA = a.match_score || a.eligibility_score || 0;
    const scoreB = b.match_score || b.eligibility_score || 0;
    return scoreB - scoreA;
  });

  const getPhaseColor = (phase?: string) => {
    if (!phase) return "bg-gray-100 text-gray-700";
    if (phase.includes("1")) return "bg-blue-100 text-blue-700";
    if (phase.includes("2")) return "bg-green-100 text-green-700";
    if (phase.includes("3")) return "bg-purple-100 text-purple-700";
    if (phase.includes("4")) return "bg-orange-100 text-orange-700";
    return "bg-gray-100 text-gray-700";
  };

  const getStatusColor = (status?: string) => {
    if (!status) return "bg-gray-100 text-gray-700";
    const statusLower = status.toLowerCase();
    if (statusLower.includes("recruit")) return "bg-green-100 text-green-700";
    if (statusLower.includes("active")) return "bg-blue-100 text-blue-700";
    if (statusLower.includes("completed"))
      return "bg-gray-100 text-gray-700";
    if (statusLower.includes("terminated"))
      return "bg-red-100 text-red-700";
    return "bg-gray-100 text-gray-700";
  };

  const getMatchScoreColor = (score: number) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-gray-600";
  };

  if (!trials || trials.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          🔬 Clinical Trials
        </h2>
        <p className="text-gray-500">
          No clinical trials matched for this patient
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          🔬 Clinical Trials
          <span className="text-sm font-normal text-gray-500">
            ({trials.length} eligible)
          </span>
        </h2>
      </div>

      <div className="space-y-4">
        {sortedTrials.map((trial) => {
          const matchScore = trial.match_score || trial.eligibility_score || 0;
          const isExpanded = expandedTrial === trial.nct_id;

          return (
            <div
              key={trial.nct_id}
              className="border rounded-lg hover:shadow-md transition-shadow"
            >
              <div
                className="p-4 cursor-pointer"
                onClick={() =>
                  setExpandedTrial(isExpanded ? null : trial.nct_id)
                }
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {trial.title || trial.brief_title || "Untitled Trial"}
                    </h3>
                    <div className="flex gap-2 mt-2 flex-wrap">
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded ${getPhaseColor(
                          trial.phase
                        )}`}
                      >
                        {trial.phase || "Phase N/A"}
                      </span>
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded ${getStatusColor(
                          trial.status
                        )}`}
                      >
                        {trial.status || "Status Unknown"}
                      </span>
                      <span className="px-2 py-1 text-xs bg-gray-100 text-gray-700 font-semibold rounded">
                        NCT{trial.nct_id}
                      </span>
                    </div>
                  </div>
                  <div className="ml-4 text-right">
                    <div
                      className={`text-2xl font-bold ${getMatchScoreColor(
                        matchScore
                      )}`}
                    >
                      {matchScore}
                    </div>
                    <div className="text-xs text-gray-500">Match Score</div>
                  </div>
                </div>

                {trial.condition && (
                  <div className="mt-2 text-sm text-gray-600">
                    <strong>Condition:</strong> {trial.condition}
                  </div>
                )}

                {trial.intervention && (
                  <div className="mt-1 text-sm text-gray-600">
                    <strong>Intervention:</strong> {trial.intervention}
                  </div>
                )}
              </div>

              {/* Expanded Section */}
              {isExpanded && (
                <div className="border-t p-4 bg-gray-50">
                  {/* Eligibility Criteria */}
                  {trial.eligibility_criteria &&
                    trial.eligibility_criteria.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-semibold text-sm mb-2">
                          Eligibility Criteria:
                        </h4>
                        <ul className="list-disc list-inside space-y-1">
                          {trial.eligibility_criteria.map((criteria, index) => (
                            <li key={index} className="text-sm text-gray-700">
                              {criteria}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {/* Matched Criteria */}
                  {trial.matched_criteria &&
                    trial.matched_criteria.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-semibold text-sm mb-2 text-green-700">
                          ✓ Patient Matches:
                        </h4>
                        <ul className="list-disc list-inside space-y-1">
                          {trial.matched_criteria.map((criteria, index) => (
                            <li
                              key={index}
                              className="text-sm text-green-700 font-medium"
                            >
                              {criteria}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {/* Actions */}
                  <div className="flex gap-2 mt-4">
                    <a
                      href={`https://clinicaltrials.gov/study/${trial.nct_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                    >
                      View on ClinicalTrials.gov →
                    </a>
                    {onEnroll && (
                      <button
                        onClick={() => onEnroll(trial.nct_id)}
                        className="px-4 py-2 bg-green-600 text-white text-sm rounded hover:bg-green-700"
                      >
                        Enroll Patient
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="mt-6 bg-blue-50 p-4 rounded">
        <p className="text-sm text-gray-700">
          <strong>Note:</strong> These trials have been matched based on patient
          characteristics. Please review full eligibility criteria before
          enrollment. Match scores above 80 indicate strong alignment with
          patient profile.
        </p>
      </div>
    </div>
  );
}
