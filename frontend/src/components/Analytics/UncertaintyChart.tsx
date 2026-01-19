"use client";

import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface UncertaintyMetrics {
  overall_uncertainty: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
  confidence_score: number;
  uncertainty_sources: {
    source: string;
    contribution: number;
  }[];
  risk_level: "low" | "medium" | "high";
}

interface UncertaintyChartProps {
  metrics: UncertaintyMetrics;
}

export default function UncertaintyChart({ metrics }: UncertaintyChartProps) {
  const getRiskBadge = (level: string) => {
    const variants = {
      low: { color: "bg-green-100 text-green-800", icon: <TrendingDown className="h-4 w-4" /> },
      medium: { color: "bg-yellow-100 text-yellow-800", icon: <Minus className="h-4 w-4" /> },
      high: { color: "bg-red-100 text-red-800", icon: <TrendingUp className="h-4 w-4" /> },
    };
    const variant = variants[level as keyof typeof variants] || variants.medium;
    return (
      <Badge className={`${variant.color} flex items-center gap-1`}>
        {variant.icon}
        {level.toUpperCase()} RISK
      </Badge>
    );
  };

  const getUncertaintyColor = (value: number) => {
    if (value < 0.3) return "bg-green-500";
    if (value < 0.6) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle>Uncertainty Quantification</CardTitle>
            <CardDescription>
              Model confidence and uncertainty metrics
            </CardDescription>
          </div>
          {getRiskBadge(metrics.risk_level)}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overall Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
            <p className="text-3xl font-bold text-blue-600">
              {(metrics.confidence_score * 100).toFixed(1)}%
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full"
                style={{ width: `${metrics.confidence_score * 100}%` }}
              />
            </div>
          </div>

          <div className="p-4 bg-red-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Overall Uncertainty</p>
            <p className="text-3xl font-bold text-red-600">
              {(metrics.overall_uncertainty * 100).toFixed(1)}%
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${getUncertaintyColor(metrics.overall_uncertainty)}`}
                style={{ width: `${metrics.overall_uncertainty * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Uncertainty Decomposition */}
        <div>
          <h3 className="font-semibold mb-3">Uncertainty Decomposition</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Epistemic (Model Uncertainty)</span>
                <span className="font-semibold">
                  {(metrics.epistemic_uncertainty * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-purple-500 h-3 rounded-full"
                  style={{ width: `${metrics.epistemic_uncertainty * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Uncertainty due to limited training data or model knowledge
              </p>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Aleatoric (Data Uncertainty)</span>
                <span className="font-semibold">
                  {(metrics.aleatoric_uncertainty * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="bg-orange-500 h-3 rounded-full"
                  style={{ width: `${metrics.aleatoric_uncertainty * 100}%` }}
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Irreducible uncertainty inherent in the data
              </p>
            </div>
          </div>
        </div>

        {/* Uncertainty Sources */}
        <div>
          <h3 className="font-semibold mb-3">Primary Uncertainty Sources</h3>
          <div className="space-y-2">
            {metrics.uncertainty_sources.map((source, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-gray-600" />
                  <span className="text-sm">{source.source}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${getUncertaintyColor(source.contribution)}`}
                      style={{ width: `${source.contribution * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-semibold min-w-[3rem] text-right">
                    {(source.contribution * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <h3 className="font-semibold text-yellow-900 mb-2">
            Clinical Recommendations
          </h3>
          <ul className="text-sm text-yellow-800 space-y-1">
            {metrics.overall_uncertainty > 0.6 && (
              <li>• High uncertainty detected - consider multidisciplinary review</li>
            )}
            {metrics.epistemic_uncertainty > 0.5 && (
              <li>• Model uncertainty elevated - additional clinical data may help</li>
            )}
            {metrics.confidence_score < 0.7 && (
              <li>• Low confidence - human-in-the-loop review recommended</li>
            )}
            {metrics.risk_level === "high" && (
              <li>• High risk case - consult with oncology specialist</li>
            )}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
