import { FETCH_EXPERIMENTS, FETCH_CCT, FETCH_TIMELINE, FETCH_METRICS, FETCH_KERNELS, FETCH_ENSEMBLE, FETCH_COMM, TEST_FETCH_JSON, UPDATE_EXPERIMENT, UPDATE_KERNEL, UPDATE_METRIC } from "./helpers/types";
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

export const fetchMetrics = () => async (dispatch) => {
  const data = await GETWrapper("fetch_metrics");
  dispatch({
    type: FETCH_METRICS,
    payload: data,
  });
};  

export const fetchKernels = () => async (dispatch) => {
  const data = await POSTWrapper("fetch_kernels", {});
  dispatch({
    type: FETCH_KERNELS,
    payload: data,
  });
};

export const fetchEnsemble = (metric) => async (dispatch) => {
  const data = await POSTWrapper("fetch_ensemble", {metric: metric});
  dispatch({
    type: FETCH_ENSEMBLE,
    payload: data,
  });
};


export const fetchJSON = (filename) => async (dispatch) => {
  const data = await GETWrapper(`static/${filename}`);
  dispatch({
    type: TEST_FETCH_JSON,
    payload: data,
  });
};

export const fetchComm = (experiment) => async (dispatch) => {
  const data = await POSTWrapper("fetch_comm", {experiment: experiment});
  dispatch({
    type: FETCH_COMM,
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
