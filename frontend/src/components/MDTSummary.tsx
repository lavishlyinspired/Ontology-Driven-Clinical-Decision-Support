export default function MDTSummary({ summary }: { summary: string }) {
  return (
    <div className="section card">
      <h3>MDT Summary</h3>
      <p style={{ margin: "0.5rem 0 0", fontSize: "1rem" }}>{summary}</p>
    </div>
  );
}
