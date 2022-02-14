import React, { Fragment } from 'react'
import {
    Grid, Paper, Typography,
} from '@material-ui/core';


function ReuseWrapper() {
    return (
        <Fragment>  
            <Paper elevation={3}>
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    Data Reuse analysis
                </Typography>
                <Grid container item direction='column' spacing={4}>
                    <Encoding />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>
            </Paper>
        </Fragment>
    )
}

function Encoding() {
    return (
        <Fragment>
            <svg width={window.innerWidth/2} height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default ReuseWrapper;
