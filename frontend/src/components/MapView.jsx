import { MapContainer, TileLayer } from "react-leaflet";
import BookMarker from "./BookMarker.jsx";

export default function MapView({ books }) {
  return (
    <MapContainer center={[32.0853, 34.7818]} zoom={13} style={{ height: "70vh" }}>
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {books.map((book) => (
        <BookMarker key={book.id} book={book} />
      ))}
    </MapContainer>
  );
}
