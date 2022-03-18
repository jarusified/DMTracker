import { FETCH_REUSE, FETCH_COMM, FETCH_EXPERIMENTS, FETCH_CCT, FETCH_TIMELINE, FETCH_METRICS } from './helpers/types';

const initialState = {
    reuse: {},
    experiments: [],
    selected_experiment: '',
    cct: {},
    timeline: {},
    comm: {},
    runtime_metrics: {},
    transfer_metrics: {},
    atts: {},
    kernel_metrics: {},
    selected_kernel_metric: 'gst_transactions',
    kernels: [],
    metrics: [],
};

export default function Reducer(state=initialState, action){
    switch (action.type) {
        case FETCH_EXPERIMENTS:
            return {
                ...state,
                experiments: action.payload.experiments,
                selected_experiment: action.payload.experiments[0],
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
                kernel_metrics: action.payload.kernel_metrics,
                runtime_metrics: action.payload.runtime_metrics,
                transfer_metrics: action.payload.transfer_metrics,
                atts: action.payload.atts,
                kernels: action.payload.kernels,
                selected_kernel_metric: action.payload.metrics[0],
                metrics: action.payload.metrics,
            }
        default:
            return state;
    }
}