import { FETCH_REUSE } from "./helpers/types";
import { SERVER_URL } from "./helpers/utils";

async function requestWrapper(url_path, json_data) {
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

export const fetchREUSE = (dataset_name) => async (dispatch) => {
  const data = await requestWrapper("fetch_reuse", { dataset: dataset_name });
  dispatch({
    type: FETCH_REUSE,
    payload: data,
  });
};
