export default function MetricCard({ title, value, subtitle }) {
  return (
    <div className="metric-box">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      {subtitle ? <div className="small-text">{subtitle}</div> : null}
    </div>
  );
}
