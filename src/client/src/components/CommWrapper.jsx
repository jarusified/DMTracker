import React, { Fragment } from 'react'
import {
    Grid, Box, Typography,
} from '@material-ui/core';


function CommWrapper() {
    return (
        <Box sx={{ p: 1, border: '1px dashed grey' }}>
            <Typography variant='overline' style={{ fontWeight: 'bold' }} spacing={1}>
                Data movement Matrix
            </Typography>

            <Grid item>
                <Grid item >
                    <Matrix />
                </Grid>
            </Grid>
            <Grid item>
            </Grid>
        </Box>
    )
}

function Matrix() {
    return (
        <Fragment>
            <svg height={window.innerHeight/4}></svg>
        </Fragment>
    )
}


export default CommWrapper;
