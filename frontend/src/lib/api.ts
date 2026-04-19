import axios from "axios";

const defaultApiUrl =
  typeof window !== "undefined"
    ? `http://${window.location.hostname}:8000`
    : "http://localhost:8000";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL?.trim() || defaultApiUrl;

const API = axios.create({
  baseURL: API_BASE_URL,
});

export const analyzeVideoAPI = async (file: File, startTime = 0) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("start_time", String(startTime));

  const response = await API.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};

export const resolveApiAssetUrl = (path: string) => {
  if (!path) return path;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
};
