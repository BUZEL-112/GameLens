import { useState, useEffect, useCallback, useRef } from "react";

/**
 * Hook for paginated, filterable game fetching.
 * Resets on genre/q changes. Supports load-more.
 */
export function useGames(genre, q) {
  const [games, setGames] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const controllerRef = useRef(null);

  // Reset and reload when filters change
  useEffect(() => {
    // Abort any in-flight request
    if (controllerRef.current) controllerRef.current.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setGames([]);
    setPage(1);
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({ page: "1" });
    if (genre) params.set("genre", genre);
    if (q) params.set("q", q);

    fetch(`/api/games?${params.toString()}`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setGames(data.games);
        setTotal(data.total);
        setHasMore(data.hasMore);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setError(err.message);
        setLoading(false);
      });

    return () => controller.abort();
  }, [genre, q]);

  const loadMore = useCallback(() => {
    if (loadingMore || !hasMore) return;

    setLoadingMore(true);
    const nextPage = page + 1;

    const params = new URLSearchParams({ page: String(nextPage) });
    if (genre) params.set("genre", genre);
    if (q) params.set("q", q);

    fetch(`/api/games?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setGames((prev) => [...prev, ...data.games]);
        setPage(nextPage);
        setHasMore(data.hasMore);
        setTotal(data.total);
        setLoadingMore(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoadingMore(false);
      });
  }, [page, genre, q, loadingMore, hasMore]);

  const reload = useCallback(() => {
    setError(null);
    setGames([]);
    setPage(1);
    setLoading(true);

    const params = new URLSearchParams({ page: "1" });
    if (genre) params.set("genre", genre);
    if (q) params.set("q", q);

    fetch(`/api/games?${params.toString()}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setGames(data.games);
        setTotal(data.total);
        setHasMore(data.hasMore);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [genre, q]);

  return { games, total, hasMore, loading, loadingMore, error, loadMore, reload };
}
