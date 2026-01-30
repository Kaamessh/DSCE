import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const fetchOptions = async () => {
  const res = await axios.get(`${API_URL}/options`);
  return res.data;
};

export const predict = async (payload) => {
  const res = await axios.post(`${API_URL}/predict`, payload);
  return res.data;
};

export const health = async () => {
  const res = await axios.get(`${API_URL}/health`);
  return res.data;
};

export const chat = async (message, history = []) => {
  const res = await axios.post(`${API_URL}/chat`, { message, history });
  return res.data;
};
