import React, { useEffect, useState } from 'react'
import { makeStyles } from '@material-ui/core/styles';
import {
    Grid, Typography, Paper
} from '@material-ui/core';
import { useDispatch, useSelector } from "react-redux";

import Matrix from '../ui/matrix';
import { fetchJSON, fetchComm } from "../actions";

const useStyles = makeStyles((theme) => ({
	rowContainer: {
        display: "grid",
        gridAutoFlow: "column",
        alignItems: "center",
    }
}));

function CommWrapper() {
    const classes = useStyles();
    const dispatch = useDispatch();
    const comm_matrix = useSelector((store) => store.comm_matrix);
    const selectedExperiment = useSelector((store) => store.selected_experiment);
    const [ h2d, set_h2d ] = useState({});
    const [ p2p, set_p2p ] = useState({});
    const [ zc, set_zc ] = useState({});
    
    useEffect(() => {
        if(selectedExperiment !== '') {
            dispatch(fetchComm(selectedExperiment));
        }
    }, [selectedExperiment]);

    useEffect(() => {
        if(Object.keys(comm_matrix).length !== 0) {
            console.log("here:", comm_matrix);
            set_h2d(comm_matrix.H2D);
            set_p2p(comm_matrix.P2P);
            set_zc(comm_matrix.ZC);
        }
    }, [comm_matrix]);

    return (
        <Paper>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Data movement Matrix
            </Typography>

            <Grid container spacing={2}>
                <Grid item xs={4}>
                    <Matrix name={"Host-Device"} data={h2d} />
                </Grid>
                <Grid item xs={4}>
                    <Matrix name={"Peer-Peer"} data={p2p} />
                </Grid>
                <Grid item xs={4}>
                    <Matrix name={"Zero-copy"} data={zc} />
                </Grid>
            </Grid>
        </Paper>
    )
}

export default CommWrapper;
