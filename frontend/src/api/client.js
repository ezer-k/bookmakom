import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
});

export function getBooks(params = {}) {
  return api.get("/books", { params }).then((response) => response.data);
}

export function getMeta() {
  return api.get("/meta").then((response) => response.data);
}

export function uploadBooks({ file, lat, lng }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("lat", String(lat));
  formData.append("lng", String(lng));
  return api.post("/upload", formData).then((response) => response.data);
}

export default api;
