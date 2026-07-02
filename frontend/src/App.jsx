import { useState } from "react";
import Header from "./components/Header.jsx";
import ListView from "./components/ListView.jsx";
import MapView from "./components/MapView.jsx";
import UploadModal from "./components/UploadModal.jsx";

const demoBooks = [];

export default function App() {
  const [view, setView] = useState("map");
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  return (
    <main>
      <Header
        view={view}
        onViewChange={setView}
        onUpload={() => setIsUploadOpen(true)}
      />
      {view === "map" ? <MapView books={demoBooks} /> : <ListView books={demoBooks} />}
      {isUploadOpen && <UploadModal onClose={() => setIsUploadOpen(false)} />}
    </main>
  );
}
