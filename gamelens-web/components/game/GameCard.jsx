import { useState, useContext } from "react";
import { sanitize } from "@/lib/sanitize";
import { recordEvent } from "@/lib/user";
import { UserContext } from "@/pages/_app";

/**
 * GameCard — core display component.
 * Three-tier image fallback: local volume -> Steam CDN -> placeholder -> CSS initials.
 * Matches the Nexus Store glassmorphic dark card style from the design system.
 */
export default function GameCard({ game, onClick, compact = false }) {
  const userId = useContext(UserContext);
  const [fallbackLevel, setFallbackLevel] = useState(0);
  const [showInitials, setShowInitials] = useState(false);

  const imageName = sanitize(game.app_name);

  const imageSources = [
    `/images/${imageName}.jpg`,
    game.id
      ? `https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/${game.id}/header.jpg`
      : null,
    "/placeholder.jpg",
  ].filter(Boolean);

  function handleImageError() {
    const nextLevel = fallbackLevel + 1;
    if (nextLevel < imageSources.length) {
      setFallbackLevel(nextLevel);
    } else {
      setShowInitials(true);
    }
  }

  function handleClick() {
    recordEvent(userId, game.app_name, "click", 0);
    if (onClick) onClick(game);
  }

  const genres = Array.isArray(game.genres) ? game.genres.slice(0, 2) : [];
  const initials = (game.app_name || "??").slice(0, 2).toUpperCase();

  const cardWidth = compact ? "192px" : "auto";

  return (
    <div
      className="game-card"
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && handleClick()}
      style={{ width: cardWidth, cursor: "pointer" }}
      aria-label={`View ${game.app_name}`}
    >
      {/* Image */}
      <div
        style={{
          position: "relative",
          width: "100%",
          paddingBottom: compact ? "133%" : "46.7%",
          overflow: "hidden",
        }}
      >
        {showInitials ? (
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: "#1b2838",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontSize: "2rem",
                fontWeight: 800,
                color: "rgba(161,217,255,0.3)",
              }}
            >
              {initials}
            </span>
          </div>
        ) : (
          <img
            src={imageSources[fallbackLevel]}
            alt={game.app_name}
            onError={handleImageError}
            loading="lazy"
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transition: "transform 0.4s ease",
            }}
          />
        )}

        {/* Hover overlay with quick-add button (Nexus Store style) */}
        <div
          className="card-hover-overlay"
          style={{
            position: "absolute",
            inset: 0,
            background: "linear-gradient(to top, rgba(0,0,0,0.85) 0%, transparent 50%)",
            opacity: 0,
            transition: "opacity 0.25s ease",
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
            padding: "0.75rem",
          }}
        >
          <button
            style={{
              width: "100%",
              background: "var(--color-secondary)",
              color: "var(--color-on-secondary)",
              border: "none",
              borderRadius: "0.375rem",
              padding: "0.375rem 0",
              fontSize: "0.65rem",
              fontWeight: 700,
              letterSpacing: "0.04em",
              cursor: "pointer",
            }}
            onClick={(e) => {
              e.stopPropagation();
              handleClick();
            }}
          >
            VIEW GAME
          </button>
        </div>

        {/* Personalized badge for compact (rec) mode */}
        {compact && (
          <span
            style={{
              position: "absolute",
              top: "0.4rem",
              right: "0.4rem",
              background: "rgba(161,217,255,0.15)",
              backdropFilter: "blur(8px)",
              color: "var(--color-primary)",
              border: "1px solid rgba(161,217,255,0.2)",
              borderRadius: "9999px",
              fontSize: "0.55rem",
              fontWeight: 700,
              padding: "0.15rem 0.4rem",
            }}
          >
            Personalized
          </span>
        )}
      </div>

      {/* Content */}
      <div style={{ padding: "0.75rem" }}>
        <h3
          style={{
            fontWeight: 700,
            fontSize: compact ? "0.75rem" : "0.85rem",
            color: "white",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            marginBottom: "0.375rem",
          }}
          title={game.app_name}
        >
          {game.app_name}
        </h3>

        {genres.length > 0 && (
          <div
            style={{
              display: "flex",
              gap: "0.25rem",
              flexWrap: "wrap",
              marginBottom: "0.375rem",
            }}
          >
            {genres.map((g) => (
              <span key={g} className="genre-chip">
                {g}
              </span>
            ))}
          </div>
        )}

        <PriceDisplay price={game.price} />
      </div>

      <style jsx>{`
        .game-card:hover .card-hover-overlay {
          opacity: 1 !important;
        }
        .game-card:hover img {
          transform: scale(1.05);
        }
      `}</style>
    </div>
  );
}

function PriceDisplay({ price }) {
  if (price === null || price === undefined) return null;

  if (typeof price === "string" && price.toLowerCase().includes("free")) {
    return (
      <span
        style={{
          fontSize: "0.75rem",
          fontWeight: 700,
          color: "#4ade80",
        }}
      >
        Free to Play
      </span>
    );
  }

  return (
    <span
      style={{
        fontSize: "0.8rem",
        fontWeight: 700,
        color: "var(--color-secondary)",
      }}
    >
      {typeof price === "number" ? `$${price.toFixed(2)}` : price}
    </span>
  );
}
