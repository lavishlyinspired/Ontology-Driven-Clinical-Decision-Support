/**
 * Monitoring Protocol Panel Component
 * Displays treatment monitoring protocols with schedules and dose adjustment recommendations
 */

import React from "react";
import { MonitoringProtocol, LabResult } from "../lib/api";

interface MonitoringProtocolPanelProps {
  patientId?: string;
  protocol: MonitoringProtocol | null;
  labResults?: LabResult[];
  onScheduleTest?: (testDate: Date, loincCodes: string[]) => void;
}

export default function MonitoringProtocolPanel({
  patientId,
  protocol,
  labResults,
  onScheduleTest,
}: MonitoringProtocolPanelProps) {
  if (!protocol) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          📋 Monitoring Protocol
        </h2>
        <p className="text-gray-500">
          No monitoring protocol available for this patient
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        📋 Monitoring Protocol
      </h2>

      {/* Protocol Summary */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-blue-50 p-4 rounded">
          <div className="text-sm text-gray-600">Regimen</div>
          <div className="text-lg font-semibold">{protocol.regimen}</div>
        </div>
        <div className="bg-blue-50 p-4 rounded">
          <div className="text-sm text-gray-600">Monitoring Frequency</div>
          <div className="text-lg font-semibold">
            {protocol.frequency || "As indicated"}
          </div>
        </div>
      </div>

      {/* Tests to Monitor */}
      {protocol.tests_to_monitor && protocol.tests_to_monitor.length > 0 && (
        <div className="mb-6">
          <h3 className="text-md font-semibold mb-3">Labs to Monitor</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {protocol.tests_to_monitor.map((test, index) => {
              const testName =
                typeof test === "string"
                  ? test
                  : test.loinc_name || test.name || "Unknown Test";
              const testFreq =
                typeof test === "string" ? protocol.frequency : test.frequency;

              return (
                <div
                  key={index}
                  className="border rounded p-3 bg-gray-50 hover:bg-gray-100"
                >
                  <div className="font-medium text-sm">{testName}</div>
                  {testFreq && (
                    <div className="text-xs text-gray-600 mt-1">
                      Frequency: {testFreq}
                    </div>
                  )}
                  {typeof test !== "string" && test.loinc_code && (
                    <div className="text-xs text-gray-500 mt-1">
                      LOINC: {test.loinc_code}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Monitoring Schedule */}
      {protocol.schedule && protocol.schedule.length > 0 && (
        <div className="mb-6">
          <h3 className="text-md font-semibold mb-3">Monitoring Schedule</h3>
          <div className="border rounded overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Timepoint
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Tests
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {protocol.schedule.map((item, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium">
                      {item.week
                        ? `Week ${item.week}`
                        : item.day
                        ? `Day ${item.day}`
                        : `Item ${index + 1}`}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">
                      {item.tests && item.tests.length > 0
                        ? item.tests.join(", ")
                        : "See protocol"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Dose Adjustment Criteria */}
      {protocol.dose_adjustments && protocol.dose_adjustments.length > 0 && (
        <div className="mb-6">
          <h3 className="text-md font-semibold mb-3">
            Dose Adjustment Criteria
          </h3>
          <div className="space-y-2">
            {protocol.dose_adjustments.map((adjustment, index) => (
              <div
                key={index}
                className="border-l-4 border-yellow-500 bg-yellow-50 p-3"
              >
                {adjustment.parameter && adjustment.threshold && (
                  <div className="text-sm">
                    <strong>If {adjustment.parameter}</strong> {adjustment.threshold}
                  </div>
                )}
                {adjustment.action && (
                  <div className="text-sm mt-1 text-gray-700">
                    → {adjustment.action}
                  </div>
                )}
                {!adjustment.parameter &&
                  !adjustment.threshold &&
                  !adjustment.action && (
                    <div className="text-sm">
                      {typeof adjustment === "string"
                        ? adjustment
                        : "See protocol for details"}
                    </div>
                  )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expected Lab Effects */}
      <div className="mt-6 bg-gray-50 p-4 rounded">
        <h3 className="text-sm font-semibold mb-2">Notes</h3>
        <p className="text-sm text-gray-700">
          This monitoring protocol is based on {protocol.regimen} treatment
          guidelines. Regular monitoring is essential for early detection of
          toxicities and optimal dose adjustments.
        </p>
        <p className="text-sm text-gray-600 mt-2">
          Created:{" "}
          {protocol.created_date
            ? new Date(protocol.created_date).toLocaleDateString()
            : "N/A"}
        </p>
      </div>
    </div>
  );
}
