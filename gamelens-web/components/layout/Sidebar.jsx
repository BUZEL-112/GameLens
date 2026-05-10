import Link from "next/link";
import { useRouter } from "next/router";

const GENRE_LINKS = [
  { href: "/?genre=Action", label: "Action", icon: "sports_esports" },
  { href: "/?genre=RPG", label: "RPG", icon: "auto_fix_high" },
  { href: "/?genre=Strategy", label: "Strategy", icon: "grid_view" },
  { href: "/?genre=Indie", label: "Indie", icon: "rocket_launch" },
  { href: "/?genre=Adventure", label: "Adventure", icon: "explore" },
  { href: "/?genre=Simulation", label: "Simulation", icon: "apartment" },
  { href: "/?genre=Sports", label: "Sports", icon: "sports_soccer" },
];

export default function Sidebar({ selectedGenre }) {
  const router = useRouter();

  function isActive(href) {
    if (href === "/") return router.pathname === "/" && !selectedGenre;
    const url = new URL(href, "http://x");
    const g = url.searchParams.get("genre");
    return selectedGenre === g;
  }

  return (
    <aside className="sidebar">
      {/* Categories section */}
      <div style={{ padding: "1.5rem 1rem 0.5rem" }}>
        <p
          style={{
            fontSize: "0.6rem",
            fontWeight: 700,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "#4ade80",
            marginBottom: "0.25rem",
          }}
        >
          CATEGORIES
        </p>
        <p style={{ fontSize: "0.6rem", color: "#475569", letterSpacing: "0.02em" }}>
          Explore by genre
        </p>
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: "0.125rem", padding: "0.5rem 0" }}>
        {GENRE_LINKS.map(({ href, label, icon }) => {
          const active = isActive(href);
          return (
            <Link
              key={label}
              href={href}
              className={`sidebar-link${active ? " active" : ""}`}
            >
              <span
                className="material-symbols-outlined"
                style={{ fontSize: "1.2rem" }}
              >
                {icon}
              </span>
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Divider */}
      <div
        style={{
          height: "1px",
          background: "rgba(255,255,255,0.05)",
          margin: "1rem 0.5rem",
        }}
      />

      {/* Friends section label */}
      <div style={{ padding: "0 1rem 0.5rem" }}>
        <p
          style={{
            fontSize: "0.6rem",
            fontWeight: 700,
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "#475569",
            marginBottom: "0.25rem",
          }}
        >
          SOCIAL
        </p>
      </div>

      <Link href="/search" className="sidebar-link">
        <span className="material-symbols-outlined" style={{ fontSize: "1.2rem" }}>
          group
        </span>
        Friends Activity
      </Link>
    </aside>
  );
}
