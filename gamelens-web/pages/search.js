import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/router";
import Head from "next/head";
import PageWrapper from "@/components/layout/PageWrapper";
import SearchBar from "@/components/filters/SearchBar";
import GenreFilter from "@/components/filters/GenreFilter";
import GameGrid from "@/components/game/GameGrid";

export default function SearchPage() {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedGenre, setSelectedGenre] = useState(null);

  // Sync from URL on mount
  useEffect(() => {
    if (!router.isReady) return;
    if (router.query.q) setSearchQuery(router.query.q);
    if (router.query.genre) setSelectedGenre(router.query.genre);
  }, [router.isReady, router.query.q, router.query.genre]);

  const handleSearch = useCallback(
    (q) => {
      setSearchQuery(q);
      const query = { ...router.query };
      if (q) query.q = q;
      else delete query.q;
      router.push({ pathname: "/search", query }, undefined, { shallow: true });
    },
    [router]
  );

  const handleGenreChange = useCallback(
    (genre) => {
      setSelectedGenre(genre);
      const query = { ...router.query };
      if (genre) query.genre = genre;
      else delete query.genre;
      router.push({ pathname: "/search", query }, undefined, { shallow: true });
    },
    [router]
  );

  return (
    <>
      <Head>
        <title>
          {searchQuery ? `"${searchQuery}" — GameLens` : "Browse Games — GameLens"}
        </title>
        <meta
          name="description"
          content="Browse and search the GameLens catalog. Filter by genre, search by name, and discover your next game."
        />
      </Head>

      <PageWrapper>
        {/* Page header */}
        <div style={{ marginBottom: "2rem" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              marginBottom: "0.75rem",
            }}
          >
            <span
              className="material-symbols-outlined"
              style={{ color: "var(--color-primary)", fontSize: "1.75rem" }}
            >
              grid_view
            </span>
            <h1
              style={{
                fontSize: "2rem",
                fontWeight: 800,
                color: "white",
                letterSpacing: "-0.02em",
              }}
            >
              {searchQuery ? `Results for "${searchQuery}"` : "Browse Games"}
            </h1>
          </div>

          {/* Search bar */}
          <div style={{ maxWidth: "480px", marginBottom: "1.5rem" }}>
            <SearchBar onSearch={handleSearch} initialValue={searchQuery} />
          </div>
        </div>

        {/* Genre filter */}
        <GenreFilter
          selectedGenre={selectedGenre}
          onGenreChange={handleGenreChange}
        />

        {/* Results grid */}
        <GameGrid genre={selectedGenre} q={searchQuery} />
      </PageWrapper>
    </>
  );
}
