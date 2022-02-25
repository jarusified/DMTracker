import { FETCH_REUSE, FETCH_TRACE, FETCH_COMM, FETCH_EXPERIMENTS } from './helpers/types';

const initialState = {
    reuse: {},
    experiments: ['a'],
    selectedExperiment: ''
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
        default:
            return state;
    }
}