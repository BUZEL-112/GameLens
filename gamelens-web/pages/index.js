import { useState, useCallback } from "react";
import Head from "next/head";
import PageWrapper from "@/components/layout/PageWrapper";
import SearchBar from "@/components/filters/SearchBar";
import GenreFilter from "@/components/filters/GenreFilter";
import GameGrid from "@/components/game/GameGrid";
import PersonalizedRow from "@/components/recommendations/PersonalizedRow";

export default function HomePage() {
  const [selectedGenre, setSelectedGenre] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  const handleSearch = useCallback((q) => setSearchQuery(q), []);
  const handleGenreChange = useCallback((genre) => setSelectedGenre(genre), []);

  return (
    <>
      <Head>
        <title>GameLens — Discover Games</title>
        <meta
          name="description"
          content="Discover your next favorite game. Powered by a Two-Tower recommendation engine trained on real player behavior."
        />
      </Head>

      <PageWrapper>
        {/* Hero Banner */}
        <section
          style={{
            position: "relative",
            height: "420px",
            borderRadius: "1rem",
            overflow: "hidden",
            marginBottom: "3rem",
            boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
          }}
        >
          {/* Background gradient hero */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "linear-gradient(135deg, #0f1923 0%, #1b2838 40%, #0a1628 100%)",
            }}
          />
          {/* Decorative glow orb */}
          <div
            style={{
              position: "absolute",
              top: "-40px",
              right: "10%",
              width: "400px",
              height: "400px",
              borderRadius: "9999px",
              background:
                "radial-gradient(circle, rgba(102,192,244,0.12) 0%, transparent 70%)",
              pointerEvents: "none",
            }}
          />
          <div
            style={{
              position: "absolute",
              bottom: "-60px",
              left: "20%",
              width: "300px",
              height: "300px",
              borderRadius: "9999px",
              background:
                "radial-gradient(circle, rgba(186,231,45,0.08) 0%, transparent 70%)",
              pointerEvents: "none",
            }}
          />

          {/* Hero vignette */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background:
                "linear-gradient(to top, rgba(19,19,19,1) 0%, rgba(19,19,19,0) 60%)",
            }}
          />

          {/* Content */}
          <div
            style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              padding: "3rem",
            }}
          >
            <span
              style={{
                display: "inline-block",
                padding: "0.25rem 0.75rem",
                borderRadius: "9999px",
                background: "var(--color-secondary)",
                color: "var(--color-on-secondary)",
                fontSize: "0.65rem",
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                marginBottom: "1rem",
              }}
            >
              AI-POWERED DISCOVERY
            </span>

            <h1
              style={{
                fontSize: "clamp(2rem, 4vw, 3.5rem)",
                fontWeight: 800,
                lineHeight: 1.1,
                letterSpacing: "-0.02em",
                color: "white",
                marginBottom: "1rem",
                maxWidth: "700px",
              }}
            >
              Discover Your Next{" "}
              <span
                style={{
                  background:
                    "linear-gradient(90deg, #66c0f4 0%, #a1d9ff 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                }}
              >
                Favorite Game
              </span>
            </h1>

            <p
              style={{
                fontSize: "1rem",
                color: "var(--color-on-surface-variant)",
                marginBottom: "1.5rem",
                maxWidth: "500px",
              }}
            >
              Powered by a Two-Tower recommendation engine trained on real Steam
              player behavior across thousands of titles.
            </p>

            {/* Search bar in hero */}
            <div style={{ maxWidth: "480px" }}>
              <SearchBar onSearch={handleSearch} />
            </div>
          </div>
        </section>

        {/* Personalized Recommendations Row */}
        <section style={{ marginBottom: "2rem" }}>
          <PersonalizedRow />
        </section>

        {/* Genre filter + Game grid */}
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
              style={{ color: "var(--color-secondary)", fontSize: "1.75rem" }}
            >
              trending_up
            </span>
            <h2 className="section-title" style={{ fontSize: "1.75rem" }}>
              Trending Now
            </h2>
          </div>

          <GenreFilter
            selectedGenre={selectedGenre}
            onGenreChange={handleGenreChange}
          />

          <GameGrid genre={selectedGenre} q={searchQuery} />
        </section>
      </PageWrapper>
    </>
  );
}
