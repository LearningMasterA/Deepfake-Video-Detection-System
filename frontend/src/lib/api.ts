import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

export const analyzeVideoAPI = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await API.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  return response.data;
};