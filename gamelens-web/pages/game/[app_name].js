import { useEffect, useContext } from "react";
import Head from "next/head";
import { getGame, getAllAppNames } from "@/lib/games";
import { recordEvent } from "@/lib/user";
import { UserContext } from "@/pages/_app";
import PageWrapper from "@/components/layout/PageWrapper";
import GameDetailHero from "@/components/game/GameDetailHero";
import SimilarGamesRow from "@/components/recommendations/SimilarGamesRow";
import SpecTag from "@/components/ui/SpecTag";

export async function getStaticPaths() {
  const names = getAllAppNames();
  const paths = names.map((name) => ({
    params: { app_name: name },
  }));
  return { paths, fallback: "blocking" };
}

export async function getStaticProps({ params }) {
  const game = getGame(decodeURIComponent(params.app_name));
  if (!game) return { notFound: true };
  return { props: { game } };
}

export default function GameDetailPage({ game }) {
  const userId = useContext(UserContext);

  useEffect(() => {
    if (userId && game) {
      recordEvent(userId, game.app_name, "impression", 0);
    }
  }, [userId, game]);

  if (!game) return null;

  const tags = Array.isArray(game.tags) ? game.tags.slice(0, 20) : [];
  const specs = Array.isArray(game.specs) ? game.specs : [];

  return (
    <>
      <Head>
        <title>{game.app_name} — GameLens</title>
        <meta
          name="description"
          content={`${game.app_name} — ${
            Array.isArray(game.genres) ? game.genres.join(", ") : ""
          }. Discover similar games and get personalized recommendations.`}
        />
      </Head>

      {/* Hero — full-bleed from sidebar edge, outside PageWrapper */}
      <div className="detail-hero-wrapper">
        <GameDetailHero game={game} />
      </div>

      <PageWrapper>
        {/* Specs section */}
        {specs.length > 0 && (
          <section
            style={{
              maxWidth: "900px",
              marginBottom: "2.5rem",
            }}
          >
            <h2
              style={{
                fontSize: "1.25rem",
                fontWeight: 700,
                color: "white",
                marginBottom: "1rem",
              }}
            >
              Features
            </h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {specs.map((spec) => (
                <SpecTag key={spec} label={spec} />
              ))}
            </div>
          </section>
        )}

        {/* Tags section */}
        {tags.length > 0 && (
          <section
            style={{
              maxWidth: "900px",
              marginBottom: "2.5rem",
            }}
          >
            <h2
              style={{
                fontSize: "1.25rem",
                fontWeight: 700,
                color: "white",
                marginBottom: "1rem",
              }}
            >
              Player Tags
            </h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {tags.map((tag) => (
                <span
                  key={tag}
                  style={{
                    padding: "0.2rem 0.65rem",
                    borderRadius: "0.25rem",
                    fontSize: "0.7rem",
                    fontWeight: 600,
                    color: "var(--color-on-surface-variant)",
                    background: "var(--color-surface-container-high)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    transition: "background 0.15s",
                    cursor: "default",
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </section>
        )}

        {/* More Like This */}
        <section style={{ marginTop: "1rem" }}>
          <SimilarGamesRow appName={game.app_name} />
        </section>
      </PageWrapper>
    </>
  );
}
