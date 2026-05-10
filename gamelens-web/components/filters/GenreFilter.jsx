import useSWR from "swr";

const fetcher = (url) => fetch(url).then((r) => r.json());

export default function GenreFilter({ selectedGenre, onGenreChange }) {
  const { data } = useSWR("/api/genres", fetcher);
  const genres = data?.genres || data || [];

  const topGenres = Array.isArray(genres) ? genres.slice(0, 12) : [];

  return (
    <div
      style={{
        display: "flex",
        gap: "0.5rem",
        flexWrap: "wrap",
        marginBottom: "1.5rem",
        paddingBottom: "1rem",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
      }}
    >
      {/* All pill */}
      <button
        onClick={() => onGenreChange(null)}
        className={`genre-chip${!selectedGenre ? " active" : ""}`}
        style={{
          cursor: "pointer",
          border: "none",
          fontSize: "0.7rem",
          padding: "0.3rem 0.75rem",
        }}
        aria-pressed={!selectedGenre}
      >
        All
      </button>

      {topGenres.map(({ genre, count }) => (
        <button
          key={genre}
          onClick={() =>
            onGenreChange(genre === selectedGenre ? null : genre)
          }
          className={`genre-chip${selectedGenre === genre ? " active" : ""}`}
          style={{
            cursor: "pointer",
            border: "none",
            fontSize: "0.7rem",
            padding: "0.3rem 0.75rem",
          }}
          aria-pressed={selectedGenre === genre}
          title={`${genre} (${count})`}
        >
          {genre}
        </button>
      ))}
    </div>
  );
}
