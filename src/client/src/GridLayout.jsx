import React from 'react'
import { makeStyles } from '@material-ui/core/styles';
import { Box, Grid } from '@mui/material';

import CommWrapper from './components/CommWrapper';
import CCTWrapper from './components/CCTWrapper';
import ReuseWrapper from './components/ReuseWrapper';
import MetricsWrapper from './components/MetricsWrapper';
import TimelineWrapper from './components/TimelineWrapper';

const useStyles = makeStyles((theme) => ({
    contentContainer: {
        padding: theme.spacing(1),
        height: '100%',
        flexGrow: 1,
        alignItems: 'stretch',
        flexWrap: 'wrap',
    },
}))

export default function GridLayout() {
    const classes = useStyles();
    return (
        <Box>
            <Grid container>
                <Grid container item direction='column'>
                    <Grid container item sm md lg xl spacing={1}>
                        <CommWrapper />
                        <CCTWrapper />
                    </Grid>
                    <Grid container item sm md lg xl spacing={1}>
                        <ReuseWrapper />
                        <MetricsWrapper />
                    </Grid>
                    <Grid container item sm md lg xl spacing={1}>
                        <TimelineWrapper />
                    </Grid>
                </Grid>
            </Grid>
        </Box>
    )
}
