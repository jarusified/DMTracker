import React, {Fragment} from 'react'
import { makeStyles } from '@material-ui/core/styles';
import { Grid } from '@material-ui/core';

const useStyles = makeStyles((theme) => ({
    contentContainer: {
        padding: theme.spacing(1),
        height: '100%',
        flexGrow: 1,
        alignItems: 'stretch',
        flexWrap: 'nowrap',
    },
}))

export default function SummaryWrapper() {
    const classes = useStyles();
    
    return (
        <Fragment>
            <Grid container className={classes.contentContainer} spacing={1}>
                <Grid container item sm={8} md={8} lg={9} xl={9}>
                </Grid>
                <Grid container item sm md lg xl direction='column' spacing={2}>
                    <Grid container item sm md lg xl>
                    </Grid>
                </Grid>
            </Grid>
        </Fragment>
    )
}
