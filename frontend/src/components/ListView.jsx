import BookListItem from "./BookListItem.jsx";

export default function ListView({ books }) {
  return (
    <section>
      <select aria-label="Sort books" defaultValue="recent">
        <option value="recent">Recent</option>
        <option value="title">Title</option>
      </select>
      <ul>
        {books.map((book) => (
          <BookListItem key={book.id} book={book} />
        ))}
      </ul>
    </section>
  );
}
