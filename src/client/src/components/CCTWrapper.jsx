import React, { Fragment } from 'react'
import {
    Grid, Paper, Typography,
} from '@material-ui/core';


function CCTWrapper() {
    return (
        <Fragment>  
            <Paper elevation={3}>
                <Typography variant='overline' style={{ fontWeight: 'bold' }}>
                    Calling Context Tree
                </Typography>
                <Grid container item direction='column' spacing={2}>
                    <CCT />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>
            </Paper>
        </Fragment>
    )
}

function CCT() {
    return (
        <Fragment>
            <svg width={window.innerWidth/2} height={window.innerHeight/3}></svg>
        </Fragment>
    )
}


export default CCTWrapper;
