import { useContext, useState, useEffect } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import { UserContext } from "@/pages/_app";
import { useRecommendations } from "@/hooks/useRecommendations";
import PageWrapper from "@/components/layout/PageWrapper";
import GameCard from "@/components/game/GameCard";
import GameCardSkeleton from "@/components/game/GameCardSkeleton";

export default function UserProfilePage() {
  const router = useRouter();
  const { user_id } = router.query;
  const currentUserId = useContext(UserContext);
  const isOwner = Boolean(user_id && user_id === currentUserId);

  const { recommendations, loading } = useRecommendations(
    isOwner ? currentUserId : null,
    20
  );
  const [enriched, setEnriched] = useState([]);

  useEffect(() => {
    if (!recommendations || recommendations.length === 0) return;
    fetch(`/api/games?all=true`)
      .then((r) => r.json())
      .then((data) => {
        const lookup = {};
        for (const g of data.games || []) lookup[g.app_name] = g;
        setEnriched(
          recommendations.map((r) => lookup[r.item_name]).filter(Boolean)
        );
      })
      .catch(() => {});
  }, [recommendations]);

  if (!router.isReady) return null;

  return (
    <>
      <Head>
        <title>Profile — GameLens</title>
      </Head>

      <PageWrapper>
        {/* Profile header */}
        <div
          className="glass"
          style={{
            borderRadius: "1rem",
            padding: "2rem",
            marginBottom: "2.5rem",
            display: "flex",
            alignItems: "center",
            gap: "1.5rem",
          }}
        >
          {/* Avatar */}
          <div
            style={{
              width: "5rem",
              height: "5rem",
              borderRadius: "9999px",
              background:
                "linear-gradient(135deg, #66c0f4 0%, #bae72d 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.75rem",
              fontWeight: 900,
              color: "#001e2d",
              flexShrink: 0,
              border: "3px solid rgba(102,192,244,0.3)",
            }}
          >
            {user_id ? user_id.slice(6, 8).toUpperCase() : "?"}
          </div>

          <div>
            <h1
              style={{
                fontSize: "1.5rem",
                fontWeight: 700,
                color: "white",
                marginBottom: "0.25rem",
              }}
            >
              Player Profile
            </h1>
            <p
              style={{
                fontSize: "0.75rem",
                color: "var(--color-outline)",
                fontFamily: "monospace",
              }}
              suppressHydrationWarning
            >
              {user_id || "..."}
            </p>
            {isOwner && (
              <span
                style={{
                  display: "inline-block",
                  marginTop: "0.5rem",
                  padding: "0.15rem 0.5rem",
                  borderRadius: "9999px",
                  fontSize: "0.6rem",
                  fontWeight: 700,
                  letterSpacing: "0.06em",
                  background: "rgba(186,231,45,0.15)",
                  color: "var(--color-secondary)",
                  border: "1px solid rgba(186,231,45,0.3)",
                }}
              >
                YOUR PROFILE
              </span>
            )}
          </div>
        </div>

        {/* Not owner message */}
        {!isOwner && user_id && (
          <div
            className="glass"
            style={{
              borderRadius: "0.75rem",
              padding: "3rem",
              textAlign: "center",
              marginBottom: "2rem",
            }}
          >
            <span
              className="material-symbols-outlined"
              style={{
                fontSize: "3rem",
                color: "var(--color-outline)",
                display: "block",
                marginBottom: "1rem",
              }}
            >
              lock
            </span>
            <p style={{ color: "var(--color-on-surface-variant)" }}>
              This profile belongs to a different player.
            </p>
          </div>
        )}

        {/* Owner: Personalized recommendations */}
        {isOwner && (
          <>
            <section style={{ marginBottom: "3rem" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  marginBottom: "1.25rem",
                }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ color: "var(--color-primary)", fontSize: "1.5rem" }}
                >
                  auto_awesome
                </span>
                <h2
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    color: "white",
                  }}
                >
                  Your Recommendations
                </h2>
              </div>

              {loading ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fill, minmax(200px, 1fr))",
                    gap: "1.5rem",
                  }}
                >
                  {Array.from({ length: 10 }).map((_, i) => (
                    <GameCardSkeleton key={i} />
                  ))}
                </div>
              ) : enriched.length > 0 ? (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fill, minmax(200px, 1fr))",
                    gap: "1.5rem",
                  }}
                >
                  {enriched.map((game, idx) => (
                    <GameCard
                      key={`${game.app_name}-${idx}`}
                      game={game}
                      onClick={() =>
                        router.push(
                          `/game/${encodeURIComponent(game.app_name)}`
                        )
                      }
                    />
                  ))}
                </div>
              ) : (
                <div
                  className="glass"
                  style={{
                    borderRadius: "0.75rem",
                    padding: "3rem",
                    textAlign: "center",
                  }}
                >
                  <span
                    className="material-symbols-outlined"
                    style={{
                      fontSize: "3rem",
                      color: "var(--color-outline)",
                      display: "block",
                      marginBottom: "1rem",
                    }}
                  >
                    explore
                  </span>
                  <p
                    style={{
                      color: "var(--color-on-surface-variant)",
                      marginBottom: "0.5rem",
                    }}
                  >
                    No personalized recommendations yet
                  </p>
                  <p
                    style={{
                      color: "var(--color-outline)",
                      fontSize: "0.85rem",
                    }}
                  >
                    Browse and interact with games to build your profile
                  </p>
                </div>
              )}
            </section>

            {/* Play History — Coming Soon */}
            <section>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  marginBottom: "1.25rem",
                }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{ color: "var(--color-outline)", fontSize: "1.5rem" }}
                >
                  history
                </span>
                <h2
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    color: "var(--color-on-surface-variant)",
                  }}
                >
                  Play History
                </h2>
              </div>

              <div
                className="glass"
                style={{
                  borderRadius: "0.75rem",
                  padding: "3rem",
                  textAlign: "center",
                  border: "1px dashed rgba(255,255,255,0.08)",
                }}
              >
                <span
                  className="material-symbols-outlined"
                  style={{
                    fontSize: "2.5rem",
                    color: "var(--color-outline-variant)",
                    display: "block",
                    marginBottom: "0.75rem",
                  }}
                >
                  construction
                </span>
                <p
                  style={{
                    color: "var(--color-outline)",
                    fontSize: "0.85rem",
                  }}
                >
                  Coming soon — play history API endpoint in development
                </p>
              </div>
            </section>
          </>
        )}
      </PageWrapper>
    </>
  );
}
