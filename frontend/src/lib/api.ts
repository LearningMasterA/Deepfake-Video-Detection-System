import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
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
