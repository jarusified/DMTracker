import React, { Fragment, useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";
import { makeStyles } from "@material-ui/core/styles";
import {
    Grid, Box, Typography,
} from '@material-ui/core';

import { fetchMetrics } from "../actions";

const useStyles = makeStyles((theme) => ({
	
}));

function MetricsWrapper() {
    const classes = useStyles();

    const dispatch = useDispatch();
    const selectedExperiment = useSelector((store) => store.selectedExperiment);

    useEffect(() => {
        if(selectedExperiment !== '') {
            dispatch(fetchMetrics(selectedExperiment));
        }
    }, [selectedExperiment]);

    return ( 
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Ensemble Performance
            </Typography>
            <Grid item>
                <Grid item>
                    <RuntimeMetrics />
                    <TransferMetrics />
                    <ProblemSizeMetrics />
                </Grid>
            </Grid>
        </Box>
    )
}

function RuntimeMetrics() {
    const runtime_metrics = useSelector((store) => store.runtime_metrics);
    const kernel_metrics = useSelector((store) => store.kernel_metrics);
    const transfer_metrics = useSelector((store) => store.transfer_metrics);

    console.log(runtime_metrics);
    console.log(transfer_metrics);
    console.log(kernel_metrics);
    return (
        <Fragment>
            {runtime_metrics.length > 0 ? (
                runtime_metrics.map((metric) => {
                    <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                        {metric.test} = {metric.mean} {metric.unit}
                    </Typography>
                })) : (<></>)
            }
        </Fragment>
    )
}

function TransferMetrics() {
    const transfer_metrics = useSelector((store) => store.transfer_metrics);

    return (
        <Fragment>
            {transfer_metrics.length > 0 ? (
                transfer_metrics.map((metric) => {
                    <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                        {metric.test} = {metric.mean} {metric.unit}
                    </Typography>
                })) : (<></>)
            }
        </Fragment>
    )
}

function ProblemSizeMetrics() {
    const problem_size_metrics = useSelector((store) => store.problem_size_metrics);
    return (
        <Fragment>
            {/* {problem_size_metrics != undefined ? (
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    {problem_size_metrics}
                </Typography>
            ) : (<></>)} */}
        </Fragment>
    )
}

export default MetricsWrapper;
