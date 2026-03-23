import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-title">EduAdapt-AI</div>
      <div className="navbar-links">
        <Link to="/">Home</Link>
      </div>
    </nav>
  );
}
