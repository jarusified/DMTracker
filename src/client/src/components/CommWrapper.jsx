import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import {
    Grid, Typography, Paper
} from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
	rowContainer: {
        display: "grid",
        gridAutoFlow: "column",
        alignItems: "center",
    }
}));

function CommWrapper() {
    const classes = useStyles();
    return (
        <Paper>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Data movement Matrix
            </Typography>

            <Grid container spacing={2}>
                <Grid item xs={4}>
                    <Matrix name={"CPU-GPU comm"} />
                </Grid>
                <Grid item xs={4}>
                    <Matrix name={"GPU-GPU comm"} />
                </Grid>
                <Grid item xs={4}>
                    <Matrix name={"Warps-Threads"} />
                </Grid>
            </Grid>
        </Paper>
    )
}

function Matrix({ name }) {
    return (
        <Paper>
            <Typography variant="overline" style={{ fontWeight: "bold" }}>
                {name}
			</Typography>
            <svg width={200} height={250}></svg>
        </Paper>
    )
}


export default CommWrapper;
