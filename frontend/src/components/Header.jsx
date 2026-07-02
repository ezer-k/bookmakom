export default function Header({ view, onViewChange, onUpload }) {
  return (
    <header>
      <input type="search" placeholder="Search books" aria-label="Search books" />
      <button type="button" onClick={() => onViewChange(view === "map" ? "list" : "map")}>
        {view === "map" ? "List" : "Map"}
      </button>
      <button type="button" onClick={onUpload}>
        Add book
      </button>
    </header>
  );
}
