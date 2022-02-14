import React, { Fragment } from 'react'
import {
    Grid, Paper, Typography,
} from '@material-ui/core';


function CommWrapper() {
    return (
        <Fragment>  
            <Paper elevation={12} square>
                <Typography variant='overline' style={{ fontWeight: 'bold' }} spacing={1}>
                    Data movement Matrix
                </Typography>
                <Grid container item direction='column' spacing={4}>
                    <Matrix />
                </Grid>
                <Grid container item direction='column' spacing={2}>
                </Grid>
            </Paper>
        </Fragment>
    )
}

function Matrix() {
    return (
        <Fragment>
            <svg width={window.innerWidth/2} height={window.innerHeight/3}></svg>
        </Fragment>
    )
}


export default CommWrapper;
