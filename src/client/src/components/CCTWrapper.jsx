import React, { Fragment, useState, useEffect } from 'react';
import { useDispatch, useSelector } from "react-redux";

import {
    Grid, Box, Typography,
} from '@material-ui/core';

import { fetchCCT } from "../actions";

function CCTWrapper() {
    const dispatch = useDispatch();
    const selectedExperiment = useSelector((store) => store.selectedExperiment);
	const cct = useSelector((store) => store.cct);

    useEffect(() => {
        if(selectedExperiment !== '') {
            dispatch(fetchCCT(selectedExperiment));
        }
    }, [selectedExperiment])

    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Calling Context Tree
            </Typography>
            <Grid item>
                <CCT />
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function CCT() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default CCTWrapper;
