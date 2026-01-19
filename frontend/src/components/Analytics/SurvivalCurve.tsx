"use client";

import { useEffect, useRef } from "react";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface SurvivalData {
  time_points: number[];
  survival_probability: number[];
  confidence_interval_lower: number[];
  confidence_interval_upper: number[];
  risk_table?: {
    time: number;
    at_risk: number;
  }[];
}

interface SurvivalCurveProps {
  data: SurvivalData;
  title?: string;
  subtitle?: string;
}

export default function SurvivalCurve({ data, title, subtitle }: SurvivalCurveProps) {
  const chartData = {
    labels: data.time_points.map((t) => `${t} months`),
    datasets: [
      {
        label: "Survival Probability",
        data: data.survival_probability,
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 3,
        fill: false,
        tension: 0.1,
      },
      {
        label: "95% CI Upper",
        data: data.confidence_interval_upper,
        borderColor: "rgba(59, 130, 246, 0.3)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: "+1",
        tension: 0.1,
      },
      {
        label: "95% CI Lower",
        data: data.confidence_interval_lower,
        borderColor: "rgba(59, 130, 246, 0.3)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "top" as const,
      },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            let label = context.dataset.label || "";
            if (label) {
              label += ": ";
            }
            label += (context.parsed.y * 100).toFixed(1) + "%";
            return label;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          callback: function (value: any) {
            return (value * 100).toFixed(0) + "%";
          },
        },
        title: {
          display: true,
          text: "Survival Probability",
        },
      },
      x: {
        title: {
          display: true,
          text: "Time (months)",
        },
      },
    },
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title || "Kaplan-Meier Survival Curve"}</CardTitle>
        <CardDescription>
          {subtitle || "Survival probability over time with 95% confidence intervals"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-96">
          <Line data={chartData} options={options} />
        </div>

        {data.risk_table && (
          <div className="mt-6">
            <h3 className="font-semibold mb-2">Risk Table</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left">Time (months)</th>
                    <th className="px-4 py-2 text-left">Patients at Risk</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {data.risk_table.map((row, idx) => (
                    <tr key={idx}>
                      <td className="px-4 py-2">{row.time}</td>
                      <td className="px-4 py-2">{row.at_risk}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          <div className="p-3 bg-blue-50 rounded">
            <p className="text-sm text-gray-600">1-Year Survival</p>
            <p className="text-2xl font-bold text-blue-600">
              {(data.survival_probability[Math.min(12, data.time_points.length - 1)] * 100).toFixed(1)}%
            </p>
          </div>
          <div className="p-3 bg-green-50 rounded">
            <p className="text-sm text-gray-600">3-Year Survival</p>
            <p className="text-2xl font-bold text-green-600">
              {(data.survival_probability[Math.min(36, data.time_points.length - 1)] * 100).toFixed(1)}%
            </p>
          </div>
          <div className="p-3 bg-purple-50 rounded">
            <p className="text-sm text-gray-600">5-Year Survival</p>
            <p className="text-2xl font-bold text-purple-600">
              {(data.survival_probability[Math.min(60, data.time_points.length - 1)] * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
