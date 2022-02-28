import { FETCH_REUSE, FETCH_COMM, FETCH_EXPERIMENTS, FETCH_CCT, FETCH_TIMELINE, FETCH_METRICS } from './helpers/types';

const initialState = {
    reuse: {},
    experiments: [],
    selectedExperiment: '',
    cct: {},
    timeline: {},
    comm: {},
    runtime_metrics: {},
    transfer_metrics: {},
    problem_size_metrics: {},
};

export default function Reducer(state=initialState, action){
    switch (action.type) {
        case FETCH_EXPERIMENTS:
            return {
                ...state,
                experiments: action.payload.experiments,
                selectedExperiment: action.payload.experiments[0],
            }
        case FETCH_REUSE:
            return {
                ...state,
                reuse: action.payload.data,
            }
        case FETCH_COMM:
            return {
                ...state,
                comm: action.payload,
            }
        case FETCH_CCT:
            return {
                ...state,
                cct: action.payload,
            }
        case FETCH_TIMELINE:
            return {
                ...state,
                timeline: action.payload,
            }
        case FETCH_METRICS:
            return {
                ...state,
                runtime_metrics: action.payload.runtime_metrics,
                transfer_metrics: action.payload.transfer_metrics,
                problem_size_metrics: action.payload.problem_size_metrics,
            }
        default:
            return state;
    }
}