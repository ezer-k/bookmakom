export default function BookListItem({ book }) {
  return (
    <li>
      <strong>{book.title}</strong>
      <span>{book.author}</span>
    </li>
  );
}
