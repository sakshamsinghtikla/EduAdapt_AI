export default function BarMeter({ label, value, max = 1 }) {
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;

  return (
    <div style={{ marginBottom: 12 }}>
      <div className="bar-row">
        <span>{label}</span>
        <span>{typeof value === "number" ? value.toFixed(4) : value}</span>
      </div>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}
