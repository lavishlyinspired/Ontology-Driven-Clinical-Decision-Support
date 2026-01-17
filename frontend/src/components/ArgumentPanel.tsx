type ArgumentEntry = {
  treatment: string;
  arguments: string[];
};

export default function ArgumentPanel({ entries }: { entries: ArgumentEntry[] }) {
  if (entries.length === 0) return null;

  return (
    <div className="section card">
      <h3>Argumentation Highlights</h3>
      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
        {entries.map((entry) => (
          <article key={entry.treatment} className="card" style={{ padding: "1rem", minHeight: "170px" }}>
            <h4 style={{ margin: "0 0 0.5rem" }}>{entry.treatment}</h4>
            <ul style={{ paddingLeft: "1rem", margin: 0 }}>
              {entry.arguments.map((arg, idx) => (
                <li key={idx} style={{ marginBottom: "0.25rem", fontSize: "0.9rem" }}>
                  {arg}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </div>
  );
}
