/**
 * LiveEmotionGraph Component
 * ==========================
 * Renders a real-time updating line chart of emotion confidence using Recharts.
 */

import React from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { format } from "date-fns";

export default function LiveEmotionGraph({ data }) {
  // Format data for Recharts
  const chartData = data.map((d) => ({
    time: format(new Date(d.timestamp), "HH:mm:ss"),
    happy: d.all_scores.happy || 0,
    sad: d.all_scores.sad || 0,
    angry: d.all_scores.angry || 0,
    neutral: d.all_scores.neutral || 0,
  }));

  return (
    <div style={{ width: "100%", height: 300, marginTop: "20px" }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <XAxis dataKey="time" stroke="#9d9db5" fontSize={12} />
          <YAxis stroke="#9d9db5" fontSize={12} domain={[0, 1]} />
          <Tooltip 
            contentStyle={{ backgroundColor: "#12122a", borderColor: "rgba(255,255,255,0.1)", borderRadius: "8px" }}
            itemStyle={{ color: "#f0f0f5" }}
          />
          <Legend />
          <Line type="monotone" dataKey="happy" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="sad" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="angry" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="neutral" stroke="#64748b" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
