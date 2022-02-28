import { FETCH_REUSE, FETCH_COMM, FETCH_EXPERIMENTS, FETCH_CCT, FETCH_TIMELINE } from './helpers/types';

const initialState = {
    reuse: {},
    experiments: [],
    selectedExperiment: '',
    cct: {},
    timeline: {},
    comm: {},
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
        default:
            return state;
    }
}