import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import {
    Grid, Box, Typography, Paper
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
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                Data movement Matrix
            </Typography>

            <Grid className={classes.rowContainer}>
                <Grid item>
                    <Matrix name={"CPU-GPU comm"} />
                </Grid>
                <Grid item>
                    <Matrix name={"GPU-GPU comm"} />
                </Grid>
                <Grid item>
                    <Matrix name={"Warps-Threads"} />
                </Grid>
            </Grid>
        </Box>
    )
}

function Matrix({ name }) {
    return (
        <Paper>
            <Typography variant="overline" style={{ fontWeight: "bold" }}>
                {name}
			</Typography>
            <svg height={window.innerHeight/5}></svg>
        </Paper>
    )
}


export default CommWrapper;
