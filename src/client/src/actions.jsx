import { FETCH_EXPERIMENTS, FETCH_CCT, FETCH_TIMELINE, FETCH_METRICS, UPDATE_EXPERIMENT, UPDATE_KERNEL, UPDATE_METRIC } from "./helpers/types";
import { SERVER_URL } from "./helpers/utils";

async function POSTWrapper(url_path, json_data) {
	const request_context = {
		// credentials: 'include',
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(json_data),
    mode: 'cors'
	};

  const response = await fetch(`${SERVER_URL}/${url_path}`, request_context);
  const data = await response.json();
  return data;
}

async function GETWrapper(url_path) {
	const request_context = {
		// credentials: 'include',
		method: "GET",
		headers: { "Content-Type": "application/json" },
    mode: 'cors'
	};

  const response = await fetch(`${SERVER_URL}/${url_path}`, request_context);
  const data = await response.json();
  return data;
}

export const fetchExperiments = () => async (dispatch) => {
  const data = await GETWrapper("fetch_experiments");
  dispatch({
    type: FETCH_EXPERIMENTS,
    payload: data,
  });
};

export const fetchCCT = (experiment_tag) => async (dispatch) => {
  const data = await POSTWrapper("fetch_cct", {experiment: experiment_tag});
  dispatch({
    type: FETCH_CCT,
    payload: data,
  });
};

export const fetchTimeline = (experiment_tag) => async (dispatch) => {
  const data = await POSTWrapper("fetch_timeline", {experiment: experiment_tag});
  dispatch({
    type: FETCH_TIMELINE,
    payload: data,
  });
};

export const fetchMetrics = (experiment_tag) => async (dispatch) => {
  const data = await POSTWrapper("fetch_metrics", {experiment: experiment_tag});
  dispatch({
    type: FETCH_METRICS,
    payload: data,
  });
};

export const updateSelectedExperiment = (exp) => (dispatch) => {
	return dispatch({
		type: UPDATE_EXPERIMENT,
		payload: exp,
	});
};

export const updateSelectedKernel = (kernel) => (dispatch) => {
	return dispatch({
		type: UPDATE_KERNEL,
		payload: kernel,
	});
};

export const updateSelectedMetric = (metric) => (dispatch) => {
	return dispatch({
		type: UPDATE_METRIC,
		payload: metric,
	});
};
