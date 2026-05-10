import { useState, useEffect } from "react";

export default function SearchBar({ onSearch, initialValue = "" }) {
  const [inputValue, setInputValue] = useState(initialValue);

  // Debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      onSearch(inputValue.trim());
    }, 300);
    return () => clearTimeout(timer);
  }, [inputValue, onSearch]);

  // Sync if parent changes initial value (e.g. search page)
  useEffect(() => {
    setInputValue(initialValue);
  }, [initialValue]);

  return (
    <div style={{ position: "relative" }}>
      <span
        className="material-symbols-outlined"
        style={{
          position: "absolute",
          left: "0.875rem",
          top: "50%",
          transform: "translateY(-50%)",
          fontSize: "1rem",
          color: "#64748b",
          pointerEvents: "none",
          zIndex: 1,
        }}
      >
        search
      </span>

      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Search games..."
        className="search-input"
        style={{ paddingRight: inputValue ? "2.5rem" : "1rem" }}
        aria-label="Search games"
      />

      {inputValue && (
        <button
          onClick={() => {
            setInputValue("");
            onSearch("");
          }}
          style={{
            position: "absolute",
            right: "0.75rem",
            top: "50%",
            transform: "translateY(-50%)",
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "#64748b",
            display: "flex",
            alignItems: "center",
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = "white")}
          onMouseLeave={(e) => (e.currentTarget.style.color = "#64748b")}
          aria-label="Clear search"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "1rem" }}>
            close
          </span>
        </button>
      )}
    </div>
  );
}
