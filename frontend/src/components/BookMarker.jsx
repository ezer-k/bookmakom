import { Marker, Popup } from "react-leaflet";

export default function BookMarker({ book }) {
  return (
    <Marker position={[book.latitude, book.longitude]}>
      <Popup>
        <strong>{book.title}</strong>
        <br />
        {book.author}
      </Popup>
    </Marker>
  );
}
