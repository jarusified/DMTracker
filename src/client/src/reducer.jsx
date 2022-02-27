import { FETCH_REUSE, FETCH_TRACE, FETCH_COMM, FETCH_EXPERIMENTS, FETCH_CCT } from './helpers/types';

const initialState = {
    reuse: {},
    experiments: [],
    selectedExperiment: '',
    cct: {},
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
        case FETCH_TRACE:
            return {
                ...state,
                selectedOp: action.payload,
            }
        case FETCH_COMM:
            return {
                ...state,
                baseGlyphData: action.payload.base,
                targetGlyphData: action.payload.target,
            }
        case FETCH_CCT:
            return {
                ...state,
                cct: action.payload,
            }
        default:
            return state;
    }
}